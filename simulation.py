import numpy as np

class TetrisSimulator:
    def __init__(self, board_width=10, board_height=20):
        self.width = board_width
        self.height = board_height

    def is_valid_position(self, board, piece_matrix, x, y):
        """
        Checks if a piece can be placed at a given (x, y) position on the board.
        """
        piece_height, piece_width = piece_matrix.shape

        # Check if the piece is within the board boundaries
        if x < 0 or x + piece_width > self.width or \
           y < 0 or y + piece_height > self.height:
            return False

        # Check for collisions with existing blocks on the board
        for r in range(piece_height):
            for c in range(piece_width):
                if piece_matrix[r, c] != 0 and board[y + r, x + c] != 0:
                    return False
        return True

    def place_piece(self, board, piece_matrix, x, y):
        """
        Places a piece on the board at a given (x, y) position.
        Returns the new board state.
        """
        new_board = board.copy()
        piece_height, piece_width = piece_matrix.shape
        for r in range(piece_height):
            for c in range(piece_width):
                if piece_matrix[r, c] != 0:
                    new_board[y + r, x + c] = 1  # Represent all pieces with '1'
        return new_board

    def clear_lines(self, board):
        """
        Clears completed lines from the board and returns the new board and lines cleared.
        """
        new_board = board.copy()
        lines_cleared = 0

        # Get rows that are not full
        rows_to_keep = [r for r in range(self.height) if not np.all(new_board[r, :] == 1)]

        lines_cleared = self.height - len(rows_to_keep)

        if lines_cleared > 0:
            # Create new empty rows at the top
            new_rows = np.zeros((lines_cleared, self.width), dtype=np.uint8)
            # Combine new rows with the rows that were kept
            new_board = np.vstack((new_rows, new_board[rows_to_keep, :]))

        return new_board, lines_cleared
