# tic_tac_toe.py
import numpy as np
import random

class TicTacToe:
    """
    Tic-Tac-Toe game environment for Q-learning.
    Handles moves, win checking, and board state.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to empty"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 starts
        return self.get_state()

    def get_state(self):
        """Return board as a flattened string for Q-table"""
        return str(self.board.flatten())

    def get_available_actions(self):
        """Return coordinates of empty spaces"""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def make_move(self, action, player):
        row, col = action
        if self.board[row, col] == 0:
            self.board[row, col] = player
            return True
        return False

    def check_winner(self):
        # Rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]

        # Columns
        for c in range(3):
            if self.board[0, c] == self.board[1, c] == self.board[2, c] != 0:
                return self.board[0, c]

        # Diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]

        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]

        # Tie
        if not self.get_available_actions():
            return 0

        return None  # Game still going

    def display(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n  0 1 2")
        print(" -----------")
        for i in range(3):
            print(f"{i}| ", end="")
            for j in range(3):
                print(f"{symbols[self.board[i, j]]} | ", end="")
            print("\n -----------")
