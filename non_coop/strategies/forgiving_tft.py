# Strategy known as "Forgiving Tit for Tat".
# Player starts by playing C in the first move.
# Player plays D if and only if the opponent has just played D twice in a row.

def strategy(history, memory):
    action = 1
    if history.shape[1] >= 2 and history[1, -1] == 0 and history[1, -2] == 0:
        action = 0
    return action, None