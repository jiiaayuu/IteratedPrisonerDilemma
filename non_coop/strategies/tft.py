# Strategy known as "Tit for Tat".
# Player starts by playing C in the first move.
# In the rest of the match, player simply copies its opponent's previous move.

def strategy(history, memory):
    action = 1
    # Plays D if and only if the opponents has just played D:
    if history.shape[1] >= 1 and history[1, -1] == 0:
        action = 0
    return action, None