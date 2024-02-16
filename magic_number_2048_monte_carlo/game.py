from __future__ import division

import time
import os
import pickle
import math

from action import *

class Game:
    def __init__(self, cols=4, rows=4):
        self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.board = self.grid_array
        for i in range(2):
            put_new_cell(self.board)
        self.score = 0
        self.over = False

    def move(self, direction):

        score = direction_dict[direction](self.board)

        self.score += max(0, score)

        next_turn = prepare_next_turn(self.board)

        if not next_turn:
            self.over = True

    def next_state(self, state, direction):
        direction_dict = {'left': push_left, 'up': push_up, 'right': push_right, 'down': push_down}
        direction_dict[direction](state)

        next_turn = prepare_next_turn(state)

        if not next_turn:
            self.over = True
        self.move(direction)
        return self.to_state()

    def show(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j]:
                    print('%4d' % self.board[i][j])
                else:
                    print('   .')
            print()
