import copy
import queue
import sys
import time

from pydantic import BaseModel

# ------ DATA CLASSES/GLOBALS ------


class State(BaseModel):
    robot_loc: tuple[int, int]
    holding_box: bool
    box_locs: set[tuple[int, int]]
    blocked_locs: set[tuple[int, int]]
    shelf_locs: set[tuple[int, int]]

    # Modify default hash method to make data types immutable
    def __hash__(self):
        return hash(
            (
                self.robot_loc,
                self.holding_box,
                frozenset(self.box_locs),
                frozenset(self.blocked_locs),
                frozenset(self.shelf_locs),
            )
        )


class Node(BaseModel):
    depth: int
    priority: float = 0
    action: str = None
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

        robot = tuple()
        n = 0
        boxes = set()
        shelves = set()
        blocked = set()
        y = 0
        for line in file:
            x = 0
            for char in line:
                if char == "@":  # robot
                    robot = (x, y)
                    n += 1
                elif char == "#":  # blocked cell (e.g. filled shelf)
                    blocked.add((x, y))
                elif char == "|":  # empty shelf
                    shelves.add((x, y))
                elif char == "*":  # box
                    boxes.add((x, y))
                x += 1
            y += 1

    init_state = State(
        robot_loc=robot,
        holding_box=False,
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


def astar(init_state):
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

        if len(node.state.box_locs) == 0 and not node.state.holding_box:
            print_path(node)
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
    x, y = state.robot_loc

    # Expand order: pickup, place, up, right, down, left
    directions = [(x, y - 1, "U"), (x + 1, y, "R"), (x, y + 1, "D"), (x - 1, y, "L")]

    # If box can be picked up, generate state with pickup action and skip movement actions
    if state.robot_loc in state.box_locs and not state.holding_box:
        state.box_locs.remove(state.robot_loc)
        state.holding_box = True
        child = Node(state=copy.deepcopy(state), depth=node.depth + 1, action="PU", parent=node)
        children.append(child)
        return children
    # If box can be placed, generate state with place action and skip movement actions
    elif state.robot_loc in state.shelf_locs and state.holding_box:
        state.shelf_locs.remove(state.robot_loc)
        state.blocked_locs.add(state.robot_loc)
        state.holding_box = False
        child = Node(state=copy.deepcopy(state), depth=node.depth + 1, action="PL", parent=node)
        children.append(child)
        return children

    # Expand directions
    for loc in directions:
        # Skip state if location is out of bounds or blocked
        if (
            (loc[0] < 0 or loc[0] > COLS - 1)
            or (loc[1] < 0 or loc[1] > ROWS - 1)
            or ((loc[0], loc[1]) in state.blocked_locs)
        ):
            continue
        new_state = State(
            robot_loc=(loc[0], loc[1]),
            holding_box=state.holding_box,
            box_locs=state.box_locs,
            blocked_locs=state.blocked_locs,
            shelf_locs=state.shelf_locs,
        )

        if new_state in closed:
            continue

        child = Node(state=new_state, depth=node.depth + 1, action=loc[2], parent=node)
        children.append(child)

    return children


def heuristics(state: State) -> float:
    # Manhattan distance
    dsts = []
    rbt = state.robot_loc
    # When not holding box
    if not state.holding_box:
        for box in state.box_locs:
            dst = abs(rbt[0] - box[0]) + abs(rbt[1] - box[1])
            dsts.append(dst)
        return min(dsts) if len(dsts) != 0 else 0
    # When holding box
    else:
        for shelf in state.shelf_locs:
            dst = abs(rbt[0] - shelf[0]) + abs(rbt[1] - shelf[1])
            dsts.append(dst)
        return min(dsts) if len(dsts) != 0 else 0


def print_path(node: Node):
    actions = []
    while True:
        if node.action == None:
            break
        actions.append(node.action)
        node = node.parent

    if INLINE:
        print(*actions[::-1])
    else:
        for action in actions[::-1]:
            print(action)


if __name__ == "__main__":
    main()
