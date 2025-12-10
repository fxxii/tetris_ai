import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import random
import math

from tetris_gymnasium.envs.tetris import Tetris
from model import TetrisNet, device
from agent import get_all_possible_states
from utils import get_rotations
from replay_buffer import ReplayBuffer
from dqn import train_step, update_target_net
from heuristics import get_board_heuristics

# --- Hyperparameters ---
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
REPLAY_BUFFER_CAPACITY = 10000
NUM_EPISODES = 1000
TARGET_UPDATE_FREQUENCY = 10 # Update target net every 10 episodes

def select_action(state_heuristics, board, piece, policy_net, steps_done):
    """
    Selects an action using an epsilon-greedy policy.
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        # --- Exploitation: Choose the best move ---
        with torch.no_grad():
            # In our setup, the "action" is choosing the best resulting state.
            # So, we find the state with the highest Q-value.
            possible_states = get_all_possible_states(board, piece)
            if not possible_states:
                return None, None

            state_heuristics_list = list(possible_states.values())
            input_tensor = torch.FloatTensor(state_heuristics_list).to(device)

            predictions = policy_net(input_tensor)
            best_move_idx = torch.argmax(predictions).item()

            best_move = list(possible_states.keys())[best_move_idx]
            next_state_heuristics = state_heuristics_list[best_move_idx]
            return best_move, next_state_heuristics
    else:
        # --- Exploration: Choose a random move ---
        possible_states = get_all_possible_states(board, piece)
        if not possible_states:
            return None, None

        random_move = random.choice(list(possible_states.keys()))
        next_state_heuristics = possible_states[random_move]
        return random_move, next_state_heuristics

def run_training_loop():
    """
    Initializes the environment and runs the full DQN training loop.
    """
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")

    policy_net = TetrisNet().to(device)
    target_net = TetrisNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Load pre-trained model if it exists
    try:
        policy_net.load_state_dict(torch.load("tetris_dqn.pth"))
        target_net.load_state_dict(torch.load("tetris_dqn.pth"))
        print("Loaded pre-trained model.")
    except FileNotFoundError:
        print("No pre-trained model found, starting from scratch.")

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    steps_done = 0

    for i_episode in range(NUM_EPISODES):
        observation, info = env.reset()
        terminated = False

        while not terminated:
            env.render()

            # 1. Get current state
            board_from_env = observation['board']
            board_cropped = env.crop_padding(board_from_env)
            binary_board = (board_cropped > 0).astype(np.uint8)
            current_piece = env.active_tetromino

            if current_piece is None:
                action = env.actions.no_op
                observation, _, terminated, _, info = env.step(action)
                continue

            # The "current state" is before any move, so lines cleared is 0.
            agg_height, holes, bumpiness = get_board_heuristics(binary_board)
            current_state_heuristics = (agg_height, holes, bumpiness, 0)

            # 2. Select an action (epsilon-greedy)
            chosen_move, next_state_heuristics = select_action(
                current_state_heuristics, binary_board, current_piece, policy_net, steps_done
            )
            steps_done += 1

            if chosen_move is None:
                action = env.actions.no_op
                observation, _, terminated, _, info = env.step(action)
                continue

            # 3. Execute the chosen action in the environment
            target_rotation_idx, target_x_unpadded = chosen_move
            rotations = get_rotations(current_piece)

            current_rotation_idx = -1
            for i, r_matrix in enumerate(rotations):
                if np.array_equal(env.active_tetromino.matrix, r_matrix):
                    current_rotation_idx = i
                    break

            if current_rotation_idx != -1:
                num_rotations = len(rotations)
                rotations_cw = (target_rotation_idx - current_rotation_idx + num_rotations) % num_rotations
                rotations_ccw = (current_rotation_idx - target_rotation_idx + num_rotations) % num_rotations

                if rotations_cw <= rotations_ccw:
                    for _ in range(rotations_cw):
                        observation, _, terminated, _, _ = env.step(env.actions.rotate_clockwise)
                else:
                    for _ in range(rotations_ccw):
                        observation, _, terminated, _, _ = env.step(env.actions.rotate_counterclockwise)

            target_x_padded = target_x_unpadded + env.padding
            while env.x < target_x_padded:
                observation, _, terminated, _, _ = env.step(env.actions.move_right)
            while env.x > target_x_padded:
                observation, _, terminated, _, _ = env.step(env.actions.move_left)

            observation, env_reward, terminated, _, info = env.step(env.actions.hard_drop)

            # 4. Calculate custom reward
            lines_cleared = info.get('lines_cleared', 0)
            reward = 0
            if terminated:
                reward = -100  # Game over penalty
            else:
                reward += lines_cleared * 10 # Reward for clearing lines
                if lines_cleared == 4:
                    reward += 50 # Bonus for a "Tetris"

            # Penalty for creating holes
            holes_before = current_state_heuristics[1]
            holes_after = next_state_heuristics[1]
            if holes_after > holes_before:
                reward -= (holes_after - holes_before) * 10

            # 5. Store the experience in the replay buffer
            replay_buffer.push(current_state_heuristics, reward, next_state_heuristics, terminated)

            # 6. Perform a training step
            loss = train_step(policy_net, target_net, replay_buffer, optimizer, BATCH_SIZE, GAMMA)
            if loss is not None:
                if steps_done % 100 == 0:
                    print(f"Episode: {i_episode}, Step: {steps_done}, Loss: {loss:.4f}")

        # 7. Update the target network
        if i_episode % TARGET_UPDATE_FREQUENCY == 0:
            update_target_net(policy_net, target_net, TAU)
            print(f"--- Target Network Updated (Episode {i_episode}) ---")

    print("Training Complete!")

    # Save the trained model
    torch.save(policy_net.state_dict(), "tetris_dqn.pth")
    print("Saved trained model to tetris_dqn.pth")

    env.close()

if __name__ == "__main__":
    run_training_loop()
