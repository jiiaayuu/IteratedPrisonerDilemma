# The Win-Stay, Lose-Shift strategy.
# If the last move resulted in the same move from the opponent (a "win"), repeat the move
# Otherwise (a "loss"), switch moves

def strategy(history, memory):
    if history.shape[1] == 0:
        return 1, None

    last_my_move = history[0, -1]
    last_opponent_move = history[1, -1]

    if last_my_move == last_opponent_move:
        return last_my_move, None
    else:
        return 1-last_my_move, None