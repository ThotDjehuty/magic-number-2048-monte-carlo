import random
random.seed(1994)

from random import random, randint, choice
import numpy as np
from numba import jit
from numba import njit, prange
import numba

@numba.jit(nopython=False, parallel=True)
def push_left(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in prange(rows):
        i, last = 0, 0
        for j in prange(columns):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i - 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i += 1
        while i < columns:
            grid[k, i] = 0
            i += 1
    return score if moved else -1

@numba.jit(nopython=False, parallel=True)
def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in prange(rows):
        i = columns - 1
        last = 0
        for j in prange(columns - 1, -1, -1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i + 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i -= 1
        while 0 <= i:
            grid[k, i] = 0
            i -= 1
    return score if moved else -1

@numba.jit(nopython=False, parallel=True)
def push_up(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in prange(columns):
        i, last = 0, 0
        for j in prange(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i - 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i += 1
        while i < rows:
            grid[i, k] = 0
            i += 1
    return score if moved else -1

@numba.jit(nopython=False, parallel=True)
def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in prange(columns):
        i, last = rows - 1, 0
        for j in prange(rows - 1, -1, -1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i + 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i -= 1
        while 0 <= i:
            grid[i, k] = 0
            i -= 1
    return score if moved else -1

@numba.jit(nopython=False)
def push(grid, direction):
    if direction & 1:
        if direction & 2:
            score = push_down(grid)
        else:
            score = push_up(grid)
    else:
        if direction & 2:
            score = push_right(grid)
        else:
            score = push_left(grid)
    return score

@numba.jit(nopython=False, parallel=True)
def put_new_cell(grid):
    n = 0
    r = 0
    i_s = [0] * 16
    j_s = [0] * 16
    for i in prange(grid.shape[0]):
        for j in prange(grid.shape[1]):
            if not grid[i, j]:
                i_s[n] = i
                j_s[n] = j
                n += 1
    if n > 0:
        r = randint(0, n - 1)
        grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4
    return n

@numba.jit(nopython=False, parallel=True)
def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in prange(rows):
        for j in prange(columns):
            e = grid[i, j]
            if not e:
                return True
            if j and e == grid[i, j - 1]:
                return True
            if i and e == grid[i - 1, j]:
                return True
    return False

@numba.jit(nopython=False)
def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid)
    any_move = any_possible_moves(grid)

    return empties or any_move

direction_dict = {'left': push_left, 'up': push_up, 'right': push_right, 'down': push_down}