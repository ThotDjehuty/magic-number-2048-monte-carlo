"""Main module."""
from __future__ import division

import time
import os
import pickle
import math

from action import *

class MonteCarlo(object):
    def __init__(self, **kwargs):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.Q = {}
        self.N = {}
        self.C = kwargs.get('C', math.sqrt(2))
        self.max_moves = max_moves  # number of trajectory
        self.sample_width = sample_width
        self.actions = ['left', 'down', 'up', 'right']

    def get_play(self, state):
        # Causes the AI to calculate the best move from the
        # current game state and return it.

        s_0 = Node(state)

        sample_width = self.sample_width
        max_moves = self.max_moves

        def best_action(s, final=False):
            # select best child

            log_total = np.log(sum(s.N[a] for a in self.actions))

            scores = []
            for a in self.actions:
                try:
                    score = ((s.Q[a] * 1. / s.N[a]) + self.C * np.sqrt(log_total / s.N[a]))
                    scores.append(score)
                except ZeroDivisionError:
                    scores.append(-1)

            a = self.actions[np.argmax(scores)]

            return a

        @numba.jit(nopython=False, parallel=True, forceobj=True)
        def run_simulate(s):
            s0 = s

            for k in prange(max_moves):
                s = s0
                a = choice(self.actions)

                while any_possible_moves(s.board):  # and np.max(s.board) < 4096:

                    for a in self.actions:
                      if(len(s.children[a]) >= sample_width):
                        a = best_action(s)
                      else:
                        # expand more
                        unsampled_actions = [a for a in self.actions if len(s.children[a]) < sample_width]

                        # change here for weighted matrix or heuristic
                        a = choice(unsampled_actions)

                    if len(s.children[a]) == sample_width:
                        s1 = choice(s.children[a])
                    else:
                        s1 = next_state(s.board, a)

                        if s1[0] is None:
                            break

                        s1 = Node(s1[0], parent=s, action=a, score=s1[1]+s.score)

                        s.children[a].append(s1)

                    s = s1
                    pass

                # back-propagation
                delta = s.score
                while s is not None:
                    s.N[a] += 1
                    s.Q[a] += delta
                    s, a = s.parent, s.action

        run_simulate(s_0)

        return best_action(s_0, True)

def aiplay(game_id):
    game = Game()
    mcts_ai = MonteCarlo()
    game.show()

    tic = time.time()
    while not game.over:
        m = mcts_ai.get_play(game.board)
        game.move(m)

        if debug:
            game.show()
            print("max_moves:", max_moves, game_id, '----->direction:', \
                m, '----->current score:', game.score, '---->max tile:', np.max(game.board))

    elapsed_time = time.time() - tic
    score = game.score
    max_tile = np.max(game.board)

    return game.board, score, max_tile, elapsed_time


if __name__ == "__main__":
    # max_moves is number of trajectory

    max_moves = 100  # change this parameter for different values from 100 to 500 (bigger will slower but higher score)
    # change this parameter will have different result thereafter

    sample_width = 20
    file_save = 'data/ai_mcts_max_moves_' + str(max_moves) + '.hkl'

    if os.path.isfile(file_save):
        with open(file_save, 'r') as f:
            list_game_board, list_score, list_max_tile, list_elapsed_time = pickle.load(f)
    else:
        list_game_board = []
        list_elapsed_time = []
        list_score, list_max_tile = [], []
        debug = True  # will control whether printout the process or not

        for i in range(100):
            game_board, score, max_tile, elapsed_time = aiplay(i)
            list_game_board.append(game_board)
            print(i, "Total score:", score, "Max Tile:", max_tile)
            list_score.append(score)
            list_max_tile.append(max_tile)
            list_elapsed_time.append(elapsed_time)

        with open(file_save, 'w') as f:
            pickle.dump((list_game_board, list_score, list_max_tile, list_elapsed_time), f)
