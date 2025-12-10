import torch
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

def train_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma):
    """
    Performs a single training step for the DQN agent.

    Args:
        policy_net: The main Q-network that is being trained.
        target_net: A separate, periodically updated Q-network for calculating target values.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        optimizer: The optimizer for the policy network (e.g., Adam).
        batch_size (int): The number of experiences to sample from the buffer.
        gamma (float): The discount factor for future rewards.

    Returns:
        The calculated loss for this training step, or None if training did not occur.
    """
    # Sample a batch of experiences from the replay buffer
    batch = replay_buffer.sample(batch_size)
    if batch is None:
        return None # Not enough experiences in buffer to train

    states, rewards, next_states, dones = batch

    # 1. Calculate the Q-values for the current states using the policy network.
    #    The model will predict the Q-value for every possible move from each state.
    #    Since our model outputs a single Q-value for a state (board heuristics),
    #    this is the predicted value for the *best* action from that state.
    current_q_values = policy_net(states)

    # 2. Calculate the target Q-values for the next states using the target network.
    #    We use `torch.no_grad()` because we don't want to update the target network's weights here.
    with torch.no_grad():
        # The target network predicts the value of the best next state.
        next_q_values = target_net(next_states)

        # If a state is terminal (done), its next Q-value is 0.
        next_q_values[dones] = 0.0

    # 3. Compute the target Q-value using the Bellman equation: R + gamma * max_a' Q(s', a')
    target_q_values = rewards + (gamma * next_q_values)

    # 4. Calculate the loss between the current and target Q-values.
    #    Smooth L1 loss is often more stable than MSE for DQN.
    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    # 5. Perform backpropagation to update the policy network's weights.
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping can help prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def update_target_net(policy_net, target_net, tau):
    """
    Performs a soft update of the target network's weights.
    θ_target = τ*θ_policy + (1 - τ)*θ_target

    Args:
        policy_net: The main Q-network.
        target_net: The target Q-network to be updated.
        tau (float): The soft update factor.
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
    target_net.load_state_dict(target_net_state_dict)
