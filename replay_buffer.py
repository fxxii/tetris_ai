import random
import torch
from collections import deque
from model import device

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences for DQN training.
    """
    def __init__(self, capacity):
        """
        Initializes the replay buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        """
        Adds an experience to the buffer.

        Args:
            state (tuple): The state representation (heuristics).
            reward (float): The reward received.
            next_state (tuple): The next state representation (heuristics).
            done (bool): Whether the episode has terminated.
        """
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A tuple of tensors (states, rewards, next_states, dones), or None if
            the buffer does not contain enough experiences.
        """
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)

        # Convert the sampled experiences into PyTorch tensors
        states = torch.FloatTensor(states).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

        return states, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)
