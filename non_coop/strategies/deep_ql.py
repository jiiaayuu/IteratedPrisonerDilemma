# The Q-Learning strategy.

import numpy as np
import random

PAYOFFS = [[1, 5], [0, 3]]       # Payoff matrix

def strategy(history, memory):
    """
    Q-Learning strategy for Iterated Prisoner's Dilemma.

    - States: (my_last_move, opp_last_move) in {0,1}, plus a 'start' state.
    - Actions: 0=Defect, 1=Cooperate.
    - Q-table stored in memory['Q'] as a dict: Q[(state), action].
    - Epsilon-greedy action selection.
    - Q-update after each round with payoff as reward.

    Args:
        history (np.ndarray): shape (2, turns_played)
            - row 0: our past moves
            - row 1: opponent's past moves
        memory (dict or None): internal memory to store Q-table, last state, etc.

    Returns:
        (action, memory):
            action (int): 0=Defect, 1=Cooperate
            memory (dict): updated memory
    """
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.1

    # If first time, initialize memory
    if memory is None:
        memory = {
            'Q': {},
            'last_state': None,
            'last_action': None
        }
        possible_states = [None, (0,0), (0,1), (1,0), (1,1)]        # "None" is the start state
        for s in possible_states:
            for a in [0, 1]:
                memory['Q'][(s, a)] = 0.0

    Q = memory['Q']
    last_state = memory['last_state']
    last_action = memory['last_action']

    turns_played = history.shape[1]

    # If this isn't the first move, update Q based on the result of the last action
    if turns_played > 0:
        my_move = history[0, -1]
        opponent_move = history[1, -1]

        reward = PAYOFFS[my_move][opponent_move]

        new_state = (my_move, opponent_move)

        old_Q = Q[(last_state, last_action)]
        max_Q_next = max(Q[(new_state, a)] for a in [0, 1])
        Q[(last_state, last_action)] = old_Q+alpha*(reward+gamma*max_Q_next-old_Q)

        memory['Q'] = Q

    # Choose an action for this turn
    if turns_played == 0:
        state = None
    else:
        my_move = history[0, -1]
        opponent_move = history[1, -1]
        state = (my_move, opponent_move)

    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        q_vals = [(a, Q[(state, a)]) for a in [0, 1]]
        action = max(q_vals, key=lambda x: x[1])[0]

    # Save current state-action pair
    memory['last_state'] = state
    memory['last_action'] = action

    return action, memory