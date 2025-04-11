# Strategy known as "Grim Trigger".
# Player will continue to cooperate until the opponent defects.
# Player's state of mind will switch to "betrayed".
# Player will defect for the rest of the match.

def strategy(history, memory):
    betrayed = False
    if memory is not None and memory:                     # Has the memory of being betrayed
        betrayed = True
    else:
        if history.shape[1] >= 1 and history[1, -1] == 0: # Just got betrayed
            betrayed = True

    if betrayed:
        return 0, True
    else:
        return 1, False