import torch
from heuristics import get_board_heuristics
from simulation import TetrisSimulator
from utils import get_rotations

def get_all_possible_states(board, piece):
    """
    Generates all possible next board states for a given piece using the simulator.
    """
    states = {}
    simulator = TetrisSimulator(board.shape[1], board.shape[0])

    rotations = get_rotations(piece)

    for rotation_idx, rotated_matrix in enumerate(rotations):
        for x in range(simulator.width - rotated_matrix.shape[1] + 1):
            # Find the final y position for the piece
            y = 0
            while simulator.is_valid_position(board, rotated_matrix, x, y + 1):
                y += 1

            # Place the piece and clear lines
            temp_board = simulator.place_piece(board, rotated_matrix, x, y)
            final_board, lines_cleared = simulator.clear_lines(temp_board)

            # Calculate heuristics
            aggregate_height, holes, bumpiness = get_board_heuristics(final_board)

            heuristics = (aggregate_height, holes, bumpiness, lines_cleared)
            states[(rotation_idx, x)] = heuristics

    return states


def choose_best_move(model, board, piece):
    """
    Scores all possible moves and chooses the best one.
    """
    possible_states = get_all_possible_states(board, piece)

    if not possible_states:
        return None

    state_values = list(possible_states.values())

    device = next(model.parameters()).device
    input_tensor = torch.FloatTensor(state_values).to(device)

    with torch.no_grad():
        model.eval()
        predictions = model(input_tensor)

    best_move_idx = torch.argmax(predictions).item()

    return list(possible_states.keys())[best_move_idx]
