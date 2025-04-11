# The Prober strategy
# Axelrod 1984, "The Value of Provocability"
# The player plays C, D, C, C in the first 4 rounds
# If the opponent only plays C during these rounds, the player always plays D (exploits)
# If the opponent plays any D, the player plays Tit-for-Tat

import numpy as np

def strategy(history, memory):
    probing_sequence = [1, 0, 1, 1] # C, D, C, C
    turns_played = history.shape[1]
    exploit_mode = memory
    move_choice = None

    if turns_played < 4:
        move_choice = probing_sequence[turns_played]

    elif turns_played == 4:
        opponent_moves = history[1]
        if np.count_nonzero(opponent_moves-1) == 0:
            exploit_mode = True
        else:
            exploit_mode = False

    if turns_played >= 4:
        if exploit_mode:
            move_choice = 0 # Always D
        else:
            move_choice = history[1, -1] # T4T

    return move_choice, exploit_mode