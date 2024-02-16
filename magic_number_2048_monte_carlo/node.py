from actions import *

class Node(object):
    "Generic tree node."
    def __init__(self, board, parent=None, action=None, score=0):
        self.name = state_to_string(board)
        self.parent = parent
        self.action = action
        self.score = score
        self.children = {
            'left': [],
            'right': [],
            'up': [],
            'down': [],
        }
        self.Q = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.N = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.board = board

    def __str__(self):
        return self.name


def state_to_string(state):
    return '|'.join(str(state[j][i]) for j in range(4) for i in range(4))


def string_to_state(state):
    elements = [int(x) for x in state.split('|')]
    return np.array(elements).reshape((4, 4))


def next_state(state, direction):
    # Takes the game state, and the move to be applied.
    # Returns the new game state.

    new_state = np.copy(state)

    score = direction_dict[direction](new_state)

    empty = prepare_next_turn(new_state)

    if not empty:
        return None, -1

    return new_state, score

