import torch
import torch.nn as nn
import torch.optim as optim
import random

PAYOFFS = [
    [1, 5],
    [0, 3]
]

def encode_state(my_move, opp_move):
    """
    Encode (my_move, opp_move) as an integer pair in {0,1,2}:
    - 0 = Defect, 1 = Cooperate, 2 = 'None' (for first turn)
    """
    if my_move is None:
        my_encoded = 2
    else:
        my_encoded = my_move
    if opp_move is None:
        opp_encoded = 2
    else:
        opp_encoded = opp_move
    return (my_encoded, opp_encoded)

def one_hot_encode(seq):
    """
    Convert a sequence of (my_move, opp_move) pairs (each in 0..2) into one-hot vectors.
    We'll use 3 possible values for each player's move => 3*3=9 possible combos.
    """
    out = []
    for (my_m, opp_m) in seq:
        idx = 3 * my_m + opp_m
        vec = [0] * 9
        vec[idx] = 1
        out.append(vec)
    # shape = (sequence_length, 9)
    return torch.tensor(out, dtype=torch.float32)

class LSTMDQN(nn.Module):
    """
    LSTM-based Q-network:
    - Input: sequence of 9D one-hot vectors (representing states).
    - We feed them through an LSTM, then the final hidden output
      goes to a linear layer of size 2 (for Q(s,Defect), Q(s,Coop)).
    """
    def __init__(self, input_dim=9, hidden_dim=16, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_c=None):
        """
        x shape: (batch_size, seq_len, input_dim=9)
        h_c: optional (h0, c0) for hidden/cell states if continuing from previous
        returns: Q-values for the final step in the sequence, shape (batch_size, 2)
        """
        out, (h, c) = self.lstm(x, h_c)
        final = out[:, -1, :]  # shape: (batch_size, hidden_dim)
        qvals = self.fc(final) # shape: (batch_size, 2)
        return qvals, (h, c)

def strategy(history, memory):
    """
    LSTM DQN strategy for IPD. Minimal example, no replay buffer or advanced features.
    We re-run the entire sequence each turn.

    memory dict:
      - net: the LSTM Q-network
      - optimizer: torch optimizer
      - states_so_far: list of (my_move, opp_move) from round 0..(t-1)
      - actions_so_far: list of actions (0 or 1) we took
    """
    gamma = 0.9
    epsilon = 0.1

    # Initialize memory if needed
    if memory is None:
        net = LSTMDQN()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        memory = {
            'net': net,
            'optimizer': optimizer,
            'states_so_far': [],
            'actions_so_far': [],
        }

    net = memory['net']
    optimizer = memory['optimizer']

    turns_played = history.shape[1]

    # If we have at least one turn done, we can do a Q-update for the last transition
    if turns_played > 0:
        my_move = history[0, -1]
        opp_move = history[1, -1]
        reward = PAYOFFS[my_move][opp_move]

        # The new sequence includes the newly-encoded state
        new_seq = memory['states_so_far'][:]
        new_seq.append(encode_state(my_move, opp_move))
        # Evaluate max Q on the new sequence
        new_x = one_hot_encode(new_seq).unsqueeze(0)  # shape (1, seq_len, 9)
        net.eval()
        with torch.no_grad():
            new_qvals, _ = net(new_x)
        max_q_next = torch.max(new_qvals[0]).item()

        # The old sequence excludes the new final pair
        old_seq = memory['states_so_far'][:-1] if len(memory['states_so_far']) > 0 else []

        # --- Here's the critical skip: if old_seq is empty, skip update ---
        if len(old_seq) > 0:
            old_x = one_hot_encode(old_seq).unsqueeze(0)  # shape (1, old_len, 9)
            last_action = memory['actions_so_far'][-1]

            net.train()
            qvals_old, _ = net(old_x)
            pred_q = qvals_old[0, last_action]

            target = reward + gamma * max_q_next
            loss = (pred_q - target) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Build the current "state" for picking our next action
    if turns_played == 0:
        current_pair = encode_state(None, None)
    else:
        my_move = history[0, -1]
        opp_move = history[1, -1]
        current_pair = encode_state(my_move, opp_move)

    # Full sequence for the next action decision
    full_seq = memory['states_so_far'][:]
    full_seq.append(current_pair)

    x = one_hot_encode(full_seq).unsqueeze(0)  # shape (1, seq_len, 9)
    net.eval()
    with torch.no_grad():
        qvals, _ = net(x)
    qvals = qvals[0]  # shape (2,)

    # Epsilon-greedy selection
    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        action = int(torch.argmax(qvals).item())

    # Update memory with the new state and action
    if turns_played == 0:
        memory['states_so_far'] = [current_pair]
        memory['actions_so_far'] = [action]
    else:
        memory['states_so_far'].append(current_pair)
        memory['actions_so_far'].append(action)

    return action, memory
