import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from model import TetrisNet, device
from agent import choose_best_move
from utils import get_rotations

def run_ai_in_game():
    """
    Initializes the Tetris environment and runs the AI agent.
    """
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    model = TetrisNet().to(device)

    observation, info = env.reset()
    terminated = False

    while not terminated:
        env.render()

        # 1. Extract the current state from the environment observation
        # The environment board has padding and uses specific IDs for pieces.
        # We need to convert it to a simple binary format for our agent.
        board_from_env = observation['board']
        board_cropped = env.crop_padding(board_from_env)
        binary_board = (board_cropped > 0).astype(np.uint8)

        # Get the active piece object from the environment
        current_piece = env.active_tetromino

        if current_piece is None:
            # This can happen briefly between pieces; we can just do nothing.
            action = env.actions.no_op
            observation, reward, terminated, truncated, info = env.step(action)
            continue

        # 2. The AI decides on the best high-level move
        best_move = choose_best_move(model, binary_board, current_piece)

        if best_move is None:
            action = env.actions.no_op
            observation, reward, terminated, truncated, info = env.step(action)
            continue

        target_rotation_idx, target_x_unpadded = best_move

        # 3. Translate the high-level move into a sequence of low-level actions

        # Get all possible rotation matrices for the current piece
        rotations = get_rotations(current_piece)

        # Find the current rotation index
        current_rotation_idx = -1
        for i, r_matrix in enumerate(rotations):
            if np.array_equal(env.active_tetromino.matrix, r_matrix):
                current_rotation_idx = i
                break

        # Execute rotations to reach the target orientation
        if current_rotation_idx != -1:
            num_rotations = len(rotations)
            rotations_cw = (target_rotation_idx - current_rotation_idx + num_rotations) % num_rotations
            rotations_ccw = (current_rotation_idx - target_rotation_idx + num_rotations) % num_rotations

            if rotations_cw <= rotations_ccw:
                for _ in range(rotations_cw):
                    action = env.actions.rotate_clockwise
                    observation, reward, terminated, truncated, info = env.step(action)
                    if terminated: break
            else:
                for _ in range(rotations_ccw):
                    action = env.actions.rotate_counterclockwise
                    observation, reward, terminated, truncated, info = env.step(action)
                    if terminated: break
            if terminated: continue

        # Execute horizontal movements to reach the target x-position
        # We must adjust the agent's target x to account for the board's padding
        target_x_padded = target_x_unpadded + env.padding

        current_x = env.x
        while current_x < target_x_padded:
            action = env.actions.move_right
            observation, reward, terminated, truncated, info = env.step(action)
            current_x = env.x
            if terminated: break
        while current_x > target_x_padded:
            action = env.actions.move_left
            observation, reward, terminated, truncated, info = env.step(action)
            current_x = env.x
            if terminated: break
        if terminated: continue

        # Finally, hard drop the piece
        action = env.actions.hard_drop
        observation, reward, terminated, truncated, info = env.step(action)

    print("Game Over!")
    env.close()

if __name__ == "__main__":
    run_ai_in_game()
