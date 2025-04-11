# The Deep Q-Learning strategy.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

PAYOFFS = [[1, 5], [0, 3]]

def state_to_index(s):
    """
    Map the 'state' to an integer index:
        None   -> 0
        (0,0)  -> 1
        (0,1)  -> 2
        (1,0)  -> 3
        (1,1)  -> 4
    """
    if s is None:
        return 0
    if s == (0, 0):
        return 1
    if s == (0, 1):
        return 2
    if s == (1, 0):
        return 3
    if s == (1, 1):
        return 4
    return 0

def index_to_onehot(idx):
    """
    One-hot encode the index into a 5D tensor.
    Example:
        0 -> [1,0,0,0,0]
        1 -> [0,1,0,0,0]
        2 -> [0,0,1,0,0]
        etc.
    """
    arr = [0]*5
    arr[idx] = 1
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

class DQN(nn.Module):
    """
    A simple 2-layer feedforward network that outputs Q-values for 2 actions.
    Input: state (5D one-hot)
    Output: Q-values [Q(state, Defect), Q(state, Cooperate)]
    """
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def strategy(history, memory):
    """
    DQN strategy for IPD.
    - One-step Q-learning update each turn, no replay buffer.
    - States: None or (my_move, opp_move) -> one-hot vector of length 5
    - Network stored in memory['net'], trained with memory['optimizer'].
    - Epsilon-greedy for exploration.

    Args:
        history (np.ndarray): shape (2, turns_played)
            - row 0: our past moves (0=D, 1=C)
            - row 1: opponent's past moves (0=D, 1=C)
        memory (dict or None): stores the DQN, last state, etc.

    Returns:
        action (int): 0=Defect, 1=Cooperate
        memory (dict): updated memory
    """
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.1

    if memory is None:
        net = DQN()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        memory = {
            'net': net,
            'optimizer': optimizer,
            'last_state': None,
            'last_action': None
        }

    net = memory['net']
    optimizer = memory['optimizer']
    last_state_idx = memory['last_state']
    last_action = memory['last_action']

    turns_played = history.shape[1]

    if turns_played > 0:
        my_move = history[0, -1]
        opp_move = history[1, -1]
        reward = PAYOFFS[my_move][opp_move]

        new_state_idx = state_to_index((my_move, opp_move))

        old_state_vec = index_to_onehot(last_state_idx)  # shape (1,5)
        new_state_vec = index_to_onehot(new_state_idx)  # shape (1,5)

        current_Q = net(old_state_vec)  # shape (1,2)
        old_Q_val = current_Q[0, last_action]

        with torch.no_grad():
            next_Q = net(new_state_vec)  # shape (1,2)
            max_Q_next = torch.max(next_Q)

        target = reward + gamma * max_Q_next.item()

        loss = (old_Q_val - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if turns_played == 0:
        state_idx = 0  # None
    else:
        my_move = history[0, -1]
        opp_move = history[1, -1]
        state_idx = state_to_index((my_move, opp_move))

    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        state_vec = index_to_onehot(state_idx)
        with torch.no_grad():
            q_vals = net(state_vec)  # shape (1,2)
        action = int(torch.argmax(q_vals, dim=1)[0].item())

    memory['last_state'] = state_idx
    memory['last_action'] = action

    return action, memory