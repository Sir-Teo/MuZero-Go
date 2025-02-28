#!/usr/bin/env python
import gym
import gym_go
import torch
import numpy as np

# Import your MuZero classes and configuration.
from main import MuZeroAgent, Config, govars  # Replace 'your_module' with your module name

# Utility function to get a human move from input.
def human_move(board_size):
    move_str = input("Enter your move as 'row col' (0-indexed) or type 'pass': ").strip().lower()
    if move_str == "pass":
        return board_size * board_size  # Last index is reserved for pass.
    try:
        row, col = map(int, move_str.split())
        if 0 <= row < board_size and 0 <= col < board_size:
            return row * board_size + col
        else:
            print("Coordinates out of bounds. Try again.")
            return human_move(board_size)
    except Exception:
        print("Invalid input format. Try again.")
        return human_move(board_size)

def main():
    # --- Setup Environment and Agent ---
    board_size = Config.board_size  # e.g., 19
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    
    # Create agent with the same hyperparameters as in training.
    agent = MuZeroAgent(board_size, Config.latent_dim, Config.max_action_size, Config.mcts_simulations)
    
    # Load trained model weights. Ensure that the file path matches your saved model.
    model_path = "muzero_model_final.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.net.load_state_dict(torch.load(model_path, map_location=device))
    agent.net.eval()

    # --- Choose Player Color ---
    color_choice = input("Choose your color (black or white): ").strip().lower()
    if color_choice not in ["black", "white"]:
        print("Invalid choice; defaulting to black.")
        color_choice = "black"
    human_color = "black" if color_choice == "black" else "white"
    agent_color = "white" if human_color == "black" else "black"
    print(f"You are playing as {human_color}. The agent will play as {agent_color}.")
    
    # --- Game Loop ---
    obs = env.reset()
    # If reset() returns a tuple, extract the observation.
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    turn = 0  # In Go, black moves first.
    
    # Optionally, print the initial board state.
    env.render()

    while not done:
        # Determine whose turn it is based on standard Go rules:
        # black (turn % 2 == 0) goes first.
        if (turn % 2 == 0 and human_color == "black") or (turn % 2 == 1 and human_color == "white"):
            # Human's turn.
            print("Your move.")
            action = human_move(board_size)
        else:
            # Agent's turn.
            print("Agent's move.")
            action, _ = agent.select_action(obs)
            print(f"Agent selects action index: {action}")
        
        # Apply the action.
        result = env.step(action)
        # Handle environments that return a 5-tuple (with truncated flag) or 4-tuple.
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            obs, reward, done, info = result
        if isinstance(obs, tuple):
            obs = obs[0]
        
        env.render()  # Render the updated board.
        turn += 1

    # --- End of Game ---
    print("Game over!")
    # You can use the reward (or another scoring method) to decide the winner.
    if reward > 0:
        print("Agent wins!")
    elif reward < 0:
        print("You win!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
