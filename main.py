import copy
import queue
import sys
import time

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
    action: str = None
    robot: int = None
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
    astar(init_state)
    tend = time.time()
    if TIME:
        print(f"Algorithm completed in {tend - tstart} seconds")


# ------ ALGORITHM ------


def astar(init_state: State):
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
                priority = child.depth + heuristics(child.state)
                child.priority = priority
                open.put(child)

    print(f"{generated} nodes generated")
    print(f"{expanded} nodes expanded")


# ------ HELPERS ------


def expand(node: Node, closed: set) -> list[Node]:
    state = copy.deepcopy(node.state)
    children = []

    for i, (x, y) in enumerate(state.robot_locs):
        # Expand order: pickup, place, up, right, down, left
        directions: list[tuple[int, int, str]] = [(x, y - 1, "U"), (x + 1, y, "R"), (x, y + 1, "D"), (x - 1, y, "L")]

        # If box can be picked up, generate state with pickup action and skip movement actions
        if (x, y) in state.box_locs and not state.holding_boxes[i]:
            state.box_locs.remove((x, y))
            state.holding_boxes[i] = True
            child = Node(state=copy.deepcopy(state), depth=node.depth + 1, action="PU", robot=i, parent=node)
            children.append(child)
            return children
        # If box can be placed, generate state with place action and skip movement actions
        elif (x, y) in state.shelf_locs and state.holding_boxes[i]:
            state.shelf_locs.remove((x, y))
            state.blocked_locs.add((x, y))
            state.holding_boxes[i] = False
            child = Node(state=copy.deepcopy(state), depth=node.depth + 1, action="PL", robot=i, parent=node)
            children.append(child)
            return children

        # Expand directions
        for new_x, new_y, action in directions:
            # Skip state if location is out of bounds or blocked
            if (
                (new_x < 0 or new_x > COLS - 1)
                or (new_y < 0 or new_y > ROWS - 1)
                or ((new_x, new_y) in state.blocked_locs)
            ):
                continue

            new_state = State(
                robot_locs=state.robot_locs,
                holding_boxes=state.holding_boxes,
                box_locs=state.box_locs,
                blocked_locs=state.blocked_locs,
                shelf_locs=state.shelf_locs,
            )
            new_state.robot_locs[i] = (x, y)

            if new_state in closed:
                continue

            child = Node(state=new_state, depth=node.depth + 1, action=action, robot=i, parent=node)
            children.append(child)

    return children


def heuristics(state: State) -> float:
    # Manhattan distance
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


def print_path(node: Node, num_robots: int):
    for i in range(num_robots):
        print(f"Robot {i+1}")
        actions = []
        while True:
            if node.action == None:  # Stop when root node is reached
                break

            if node.robot == i:  # Only add action if it belongs to current robot
                actions.append(node.action)
            node = node.parent

        if INLINE:
            print(*actions[::-1])
        else:
            for action in actions[::-1]:
                print(action)


if __name__ == "__main__":
    main()
