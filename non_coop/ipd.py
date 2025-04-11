import os
import itertools
import importlib
import numpy as np
import random
import matplotlib.pyplot as plt

STRATEGIES = 'strategies'
RESULTS = 'results.txt'
IMAGES_FOLDER = 'images'

PAYOFFS = [[1, 5], [0, 3]]       # Payoff matrix
MOVES = ["D", "C"]              # D = defect, C = cooperate

os.makedirs(IMAGES_FOLDER, exist_ok=True)

def get_player_history(full_history, player_index, current_turn):
    """
    Returns the game history visible to the specified player up to the current turn.
    If player is the opponent, flip the history for their perspective.
    """
    history = full_history[:, :current_turn].copy()
    if player_index == 1:
        history = np.flip(history, axis=0)
    return history

def normalize_move(move):
    """
    Normalizes the move to 0 (defect) or 1 (cooperate).
    """
    if isinstance(move, str):
        return 0 if move == "defect" else 1
    return int(bool(move))

def simulate_match(strategy_pair):
    """
    Runs a complete match between two strategies and records the history.
    """
    strategy_A = importlib.import_module(f"{STRATEGIES}.{strategy_pair[0]}")
    strategy_B = importlib.import_module(f"{STRATEGIES}.{strategy_pair[1]}")
    memory_A, memory_B = None, None

    game_length = int(200 - 40 * np.log(1 - random.random()))
    history = np.zeros((2, game_length), dtype=int)

    for turn in range(game_length):
        move_A, memory_A = strategy_A.strategy(get_player_history(history, 0, turn), memory_A)
        move_B, memory_B = strategy_B.strategy(get_player_history(history, 1, turn), memory_B)
        history[0, turn] = normalize_move(move_A)
        history[1, turn] = normalize_move(move_B)

    return history

def calculate_average_scores(history):
    """
    Calculates average scores per turn for both players based on game history.
    """
    total_score_A, total_score_B = 0, 0
    rounds = history.shape[1]
    score_progression_A, score_progression_B = [], []

    for turn in range(rounds):
        move_A = history[0, turn]
        move_B = history[1, turn]
        total_score_A += PAYOFFS[move_A][move_B]
        total_score_B += PAYOFFS[move_B][move_A]

        score_progression_A.append(total_score_A / (turn + 1))
        score_progression_B.append(total_score_B / (turn + 1))

    return (total_score_A / rounds, total_score_B / rounds), (score_progression_A, score_progression_B)

def plot_match(history, score_progression, strategy_pair):
    """
    Generates and saves a plot of score progression over time for a match.
    """
    rounds = history.shape[1]
    x = list(range(1, rounds + 1))
    score_A, score_B = score_progression

    plt.figure()
    plt.plot(x, score_A, label=f'{strategy_pair[0]}', linewidth=2)
    plt.plot(x, score_B, label=f'{strategy_pair[1]}', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Average Score')
    plt.title(f'Match: {strategy_pair[0]} vs. {strategy_pair[1]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_FOLDER}/{strategy_pair[0]}_vs_{strategy_pair[1]}.jpg")
    plt.close()

def write_match_results(file, strategy_pair, history, score_A, score_B):
    """
    Writes the results of a single match to the results file.
    """
    file.write(f"{strategy_pair[0]} (P1) VS. {strategy_pair[1]} (P2)\n")
    for player in range(2):
        file.write(" ".join(MOVES[move] for move in history[player]) + "\n")
    file.write(f"Final score for {strategy_pair[0]}: {score_A:.3f}\n")
    file.write(f"Final score for {strategy_pair[1]}: {score_B:.3f}\n\n")

def pad_string(text, total_length):
    """
    Pads a string with spaces to the desired total length.
    """
    return text + " " * (total_length - len(text))

def plot_final_scores(score_dict):
    """
    Generates and saves a bar chart of final tournament scores.
    """
    strategies = list(score_dict.keys())
    scores = list(score_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(strategies, scores)
    plt.ylabel('Total Score')
    plt.xlabel('Strategy')
    plt.title('Final Tournament Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{IMAGES_FOLDER}/final_tournament_scores.jpg")
    plt.close()

def run_tournament(input_folder, output_file):
    """
    Runs a full round-robin tournament between all strategy files in the input folder.
    """
    print(f"Starting tournament, reading files from {input_folder}.")
    strategy_files = [file[:-3] for file in os.listdir(input_folder) if file.endswith(".py")]
    scores = {strategy: 0 for strategy in strategy_files}

    with open(output_file, "w+") as results_file:
        for strategy_pair in itertools.combinations(strategy_files, 2):
            history = simulate_match(strategy_pair)
            (score_A, score_B), score_progression = calculate_average_scores(history)

            write_match_results(results_file, strategy_pair, history, score_A, score_B)
            plot_match(history, score_progression, strategy_pair)

            scores[strategy_pair[0]] += score_A
            scores[strategy_pair[1]] += score_B

        results_file.write("\n\nTOTAL SCORES\n")
        scores_array = np.array([scores[strategy] for strategy in strategy_files])
        rankings = np.argsort(scores_array)

        for rank in range(len(strategy_files)):
            idx = rankings[-1 - rank]
            total_score = scores_array[idx]
            average_score = total_score / (len(strategy_files) - 1)
            results_file.write(
                f"#{rank + 1}: {pad_string(strategy_files[idx] + ':', 16)} {total_score:.3f}  ({average_score:.3f} average)\n"
            )

    plot_final_scores(scores)
    print(f"Tournament completed, results and visualizations saved to {output_file} and '{IMAGES_FOLDER}' folder.")

# Run the tournament
run_tournament(STRATEGIES, RESULTS)