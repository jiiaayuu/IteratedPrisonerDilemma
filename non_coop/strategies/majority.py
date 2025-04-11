# The majority strategy.
# Axelrod 1984, "Evolution of Cooperation"
# The player counts the opponent's move so far
# If the opponent has played C at least as many times as D, the player plays C. Otherwise, the player plays D

import numpy as np

def strategy(history, memory):
    if history.shape[1] == 0:
        return 1, None

    opponent_history = history[1]
    cooperations = np.count_nonzero(opponent_history == 1)

    if cooperations >= opponent_history.size/2:
        return 1, None
    else:
        return 0, None