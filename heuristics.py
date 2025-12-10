import numpy as np

def get_board_heuristics(board):
    """
    Calculates the 'eyes' of the AI: specific metrics representing board state.
    board: 2D numpy array (20 rows x 10 cols)
    """
    rows, cols = board.shape
    heights = []

    # 1. Calculate Column Heights
    # Find the first row from the top that has a '1' for each column
    for c in range(cols):
        column = board[:, c]
        occupied = np.where(column == 1)[0]
        if len(occupied) > 0:
            heights.append(rows - occupied[0])
        else:
            heights.append(0)

    # 2. Aggregate Height (AI wants to minimize this)
    aggregate_height = sum(heights)

    # 3. Holes (Empty spaces with a block above them - AI wants to avoid)
    holes = 0
    for c in range(cols):
        has_block_above = False
        for r in range(rows):
            if board[r, c] == 1:
                has_block_above = True
            elif board[r, c] == 0 and has_block_above:
                holes += 1

    # 4. Bumpiness (Sum of differences between adjacent columns)
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])

    return aggregate_height, holes, bumpiness
