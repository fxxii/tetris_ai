import numpy as np
from tetris_gymnasium.components.tetromino import Tetromino

def get_rotations(piece: Tetromino):
    """
    Generates a list of all unique rotation matrices for a given tetromino.
    """
    rotations = []
    current_piece_matrix = piece.matrix
    for _ in range(4):
        is_unique = True
        for r_matrix in rotations:
            if np.array_equal(r_matrix, current_piece_matrix):
                is_unique = False
                break
        if is_unique:
            rotations.append(current_piece_matrix)

        # Rotate the matrix for the next iteration (counter-clockwise)
        current_piece_matrix = np.rot90(current_piece_matrix, k=1)

    return rotations
