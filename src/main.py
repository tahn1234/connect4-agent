import os
import random
from typing import Dict, Final, Optional

from agent import Agent
from config import SearchConfig
from game_state import GameState
from position_classifier import PositionClassifier

SYMBOLS: Final[Dict[int, str]] = {0: "⚪", 1: "🔴", -1: "🔵"}


def play_game(player1: Agent, player2: Agent):
    """Play a complete game between two agents"""
    state = GameState()

    while True:
        display_board(state)
        current_player = player1 if state.current_player == 1 else player2
        player_symbol = SYMBOLS[state.current_player]

        move = current_player.get_best_move(state)
        print(
            f"\nPlayer {state.current_player} {player_symbol} chooses column {move + 1}"
        )

        state.make_move(move)
        result = state.check_win()

        if result is not None:
            display_board(state)
            if result == 0:
                print("\nIt's a draw!")
            else:
                winner = f"Player {result} {SYMBOLS[result]}"
                print(f"\n{winner} wins!")
            break


def display_board(state: GameState):
    """Display the current board state"""
    print("\n")
    for row in state.board:
        print("|", end="")
        for cell in row:
            print(f"{SYMBOLS[cell]}", end="")
        print("|")
    print("\n")


def get_human_move(state: GameState) -> int:
    """Get move from human player"""
    valid_moves = state.get_valid_moves()
    while True:
        try:
            move = int(input("\nEnter column (1-7): ")) - 1
            if move in valid_moves:
                return move
            print("Invalid move! Please try again.")
        except ValueError:
            print("Please enter a valid number!")


def play_game_with_mode(
    mode: str,
    player1: Agent,
    player2: Optional[Agent] = None,
    human_starts: bool = True,
):
    """Play a game based on selected mode"""
    state = GameState()

    # For human vs agent games, determine if human is player 1 or 2
    def is_human_turn(current_player):
        return (
            human_starts
            and current_player == 1
            or not human_starts
            and current_player == -1
            if mode == "human_vs_agent"
            else False
        )

    # Display who goes first
    if mode == "human_vs_agent":
        print(f"\n{'You' if human_starts else 'AI'} will play first as {SYMBOLS[1]}")
    else:
        print(f"\nPlayer 1 {SYMBOLS[1]} will play first")

    while True:
        display_board(state)
        player_symbol = SYMBOLS[state.current_player]

        # Determine current player's move
        if mode == "agent_vs_agent":
            current_agent = player1 if state.current_player == 1 else player2
            move = current_agent.get_best_move(state)  # pyright: ignore
            print(
                f"\nPlayer {state.current_player} {player_symbol} chooses column {move + 1}"
            )

        elif mode == "human_vs_agent":
            if is_human_turn(state.current_player):
                move = get_human_move(state)
                print(f"\nYou {player_symbol} chose column {move + 1}")
            else:
                move = player1.get_best_move(state)
                print(f"\nAI {player_symbol} chooses column {move + 1}")

        # Make the move
        state.make_move(move)  # pyright: ignore
        result = state.check_win()

        # Check for game end
        if result is not None:
            display_board(state)
            if result == 0:
                print("\nIt's a draw!")
            else:
                # Determine winner message based on game mode and who played as which player
                if mode == "human_vs_agent":
                    is_human_winner = (result == 1 and human_starts) or (
                        result == -1 and not human_starts
                    )
                    winner = "You" if is_human_winner else "AI"
                else:
                    winner = f"Player {result} {SYMBOLS[result]}"
                print(f"\n{winner} wins!")
            break


def main():
    """Main function with game mode selection and random player order"""

    # Load the trained classifier
    classifier = PositionClassifier()
    model_path = "models/connect4_analyzer_final.pkl"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print("Training new classifier model...")
        classifier.train(save_path=model_path, tune_params=False, balance_data=True)

    # Create agents
    config = SearchConfig(MODEL_WEIGHT=0.7, NUM_ROLLOUTS=100)
    agent1 = Agent(classifier=classifier, config=config)
    agent2 = Agent(classifier=classifier, config=config)

    # Game mode selection
    print("\nWelcome to Connect 4!")
    print("1. Human vs AI")
    print("2. AI vs AI")

    while True:
        try:
            choice = int(input("\nSelect game mode (1 or 2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice! Please select 1 or 2.")
        except ValueError:
            print("Please enter a valid number!")

    # Randomly determine who goes first
    human_starts = random.choice([True, False])

    # Play game based on selected mode
    if choice == 1:
        play_game_with_mode("human_vs_agent", agent1, human_starts=human_starts)
    else:
        play_game_with_mode("agent_vs_agent", agent1, agent2)

    # Ask to play again
    while True:
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again in ["y", "n"]:
            if play_again == "y":
                print("\n" + "=" * 50 + "\n")
                main()  # Restart the game
            break
        print("Please enter 'y' or 'n'!")


if __name__ == "__main__":
    main()
