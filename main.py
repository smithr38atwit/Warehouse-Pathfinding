import copy
import heapq
import queue
import sys
import time
from math import floor

from pydantic import BaseModel

# ------ DATA CLASSES/GLOBALS ------


class State(BaseModel):
    robot_locs: list[tuple[int, int]]
    holding_boxes: list[bool]
    box_locs: set[tuple[int, int]]
    blocked_locs: set[tuple[int, int]]
    shelf_locs: set[tuple[int, int]]

    # Modify default hash method to make data types immutable
    def __hash__(self):
        return hash(
            (
                tuple(self.robot_locs),
                tuple(self.holding_boxes),
                frozenset(self.box_locs),
                frozenset(self.blocked_locs),
                frozenset(self.shelf_locs),
            )
        )


class Node(BaseModel):
    depth: int
    priority: float = 0
    actions: list[str] = []
    state: State
    parent: "Node" = None

    # Define Nodes "<" comparison to compare priorities
    def __lt__(self, other: "Node"):
        return self.priority < other.priority


INLINE = True
TIME = True
COLS, ROWS = (0, 0)


# ------ MAIN FUNCTION ------


def main():
    # Parse arguments
    try:
        h = sys.argv[1]
        w = float(sys.argv[2])
    except IndexError:
        print("Missing argument")
        exit(2)

    if h not in ["h0", "hman", "hman+"]:
        print("Invalid heuristic")
        exit(2)

    # Parse file input
    with sys.stdin as file:
        global COLS, ROWS
        COLS = int(file.readline().strip())
        ROWS = int(file.readline().strip())

        robots = []
        holding = []
        boxes = set()
        shelves = set()
        blocked = set()
        y = 0
        for line in file:
            x = 0
            for char in line:
                if char == "@":  # robot
                    robots.append((x, y))
                    holding.append(False)
                elif char == "#":  # blocked cell (e.g. filled shelf)
                    blocked.add((x, y))
                elif char == "|":  # empty shelf
                    shelves.add((x, y))
                elif char == "*":  # box
                    boxes.add((x, y))
                x += 1
            y += 1

    init_state = State(
        robot_locs=robots,
        holding_boxes=holding,
        box_locs=boxes,
        blocked_locs=blocked,
        shelf_locs=shelves,
    )

    tstart = time.time()
    astar(init_state, h, w)
    tend = time.time()
    if TIME:
        print(f"Algorithm completed in {tend - tstart} seconds")


# ------ ALGORITHM ------


def astar(init_state: State, h: str, w: float):
    init_node = Node(state=init_state, depth=0)
    open = queue.PriorityQueue()
    closed = set()
    open.put(init_node)
    generated = 1
    expanded = 0

    while True:
        if open.empty():
            print("Search Failed")
            break

        node: Node = open.get()
        # Prevent node from being expanded if its state has already been closed
        if node.state in closed:
            continue
        closed.add(node.state)

        if len(node.state.box_locs) == 0 and not any(node.state.holding_boxes):
            print_path(node, len(init_state.robot_locs))
            break
        else:
            children = expand(node, closed)
            expanded += 1
            generated += len(children)
            for child in children:
                priority = child.depth + w * heuristics(child.state, h)
                child.priority = priority
                open.put(child)

    print(f"{generated} nodes generated")
    print(f"{expanded} nodes expanded")


# ------ HELPERS ------


def expand(node: Node, closed: set) -> list[Node]:
    state = node.state

    # Generate all possible actions for each robot
    actions: list = []
    for i, (x, y) in enumerate(state.robot_locs):
        # Expand order: pickup/place, up, right, down, left
        directions: list[tuple[int, int, str]] = [(x, y - 1, "U"), (x + 1, y, "R"), (x, y + 1, "D"), (x - 1, y, "L")]
        robot_actions: list[tuple[str, tuple]] = []

        # If box can be picked up, generate state with pickup action and skip movement actions
        if (x, y) in state.box_locs and not state.holding_boxes[i]:
            robot_actions.append(("PU", (x, y)))
            actions.append(robot_actions)
            continue
        # If box can be placed, generate state with place action and skip movement actions
        elif (x, y) in state.shelf_locs and state.holding_boxes[i]:
            robot_actions.append(("PL", (x, y)))
            actions.append(robot_actions)
            continue

        # Expand directions
        for new_x, new_y, action in directions:
            # Skip state if location is out of bounds or blocked
            if (
                (new_x < 0 or new_x > COLS - 1)
                or (new_y < 0 or new_y > ROWS - 1)
                or ((new_x, new_y) in state.blocked_locs)
            ):
                continue

            robot_actions.append((action, (new_x, new_y)))

        actions.append(robot_actions)

    # Generate nodes from all possible action combos
    action_combos: list[list[tuple[str, tuple]]] = generate_action_combinations(len(state.robot_locs), [], actions)

    children = []
    for action_list in action_combos:
        new_state = copy.deepcopy(state)
        new_actions = []
        for i, (action, location) in enumerate(action_list):
            if action == "PU":  # pick up box
                new_state.box_locs.remove(location)
                new_state.holding_boxes[i] = True
            elif action == "PL":  # place box
                new_state.shelf_locs.remove(location)
                new_state.blocked_locs.add(location)
                new_state.holding_boxes[i] = False

            new_state.robot_locs[i] = location
            new_actions.append(action)

        if new_state in closed or len(new_state.robot_locs) != len(set(new_state.robot_locs)):
            # skip if new state already exists or if there are duplicates in robot locations (i.e. robots collide)
            continue

        child = Node(state=new_state, depth=node.depth + 1, actions=new_actions, parent=node)
        children.append(child)

    return children


def generate_action_combinations(remaining_robots: int, current_combination: list, actions: list[list[tuple]]):
    if remaining_robots == 0:
        return [current_combination]

    combinations = []
    for action in actions[len(actions) - remaining_robots]:
        new_combination = current_combination + [action]
        combinations.extend(generate_action_combinations(remaining_robots - 1, new_combination, actions))

    return combinations


def heuristics(state: State, h: str) -> float:
    match h:
        case "h0":  # h(n) = 0
            return 0
        case "hman":  # Manhattan distance
            dsts = 0
            for i, (x, y) in enumerate(state.robot_locs):
                sub_dsts = []
                # When not holding box
                if not state.holding_boxes[i]:
                    for box in state.box_locs:
                        dst = abs(x - box[0]) + abs(y - box[1])
                        sub_dsts.append(dst)
                    dsts += min(sub_dsts) if len(sub_dsts) != 0 else 0
                # When holding box
                else:
                    for shelf in state.shelf_locs:
                        dst = abs(x - shelf[0]) + abs(y - shelf[1])
                        sub_dsts.append(dst)
                    dsts += min(sub_dsts) if len(sub_dsts) != 0 else 0
            return dsts
        case "hman+":  # Manhattan distance & remaining boxes
            total_distance = 0
            distances: list[list[int]] = []
            # Get distances for each robot
            # Add array of distances to boxes to distances if not holding a box
            # Otherwise just add to total distance
            for i, (x, y) in enumerate(state.robot_locs):
                sub_dsts = []
                # When not holding box
                if not state.holding_boxes[i]:
                    for box in state.box_locs:
                        dst = abs(x - box[0]) + abs(y - box[1])
                        sub_dsts.append(dst)
                    distances.append(sub_dsts if len(sub_dsts) != 0 else [0])
                # When holding box
                else:
                    for shelf in state.shelf_locs:
                        dst = abs(x - shelf[0]) + abs(y - shelf[1])
                        sub_dsts.append(dst)
                    total_distance += min(sub_dsts) if len(sub_dsts) != 0 else 0

            # Get shortest distances to boxes for each robot, splitting boxes as evenly as possible
            for dsts in distances:
                n = floor(len(state.box_locs) / len(distances))
                total_distance += sum(heapq.nsmallest(n, dsts))

            return total_distance


def print_path(node: Node, num_robots: int):
    for i in range(num_robots):
        actions = []
        init_loc = tuple()
        temp_node = node
        while True:
            if len(temp_node.actions) == 0:  # Stop when root node is reached
                init_loc = temp_node.state.robot_locs[i]
                break

            actions.append(temp_node.actions[i])
            temp_node = temp_node.parent

        print(f"Robot {i+1} {init_loc}:")
        if INLINE:
            print(*actions[::-1])
        else:
            for action in actions[::-1]:
                print(action)
        print()


if __name__ == "__main__":
    main()
