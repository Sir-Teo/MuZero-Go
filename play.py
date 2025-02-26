import gym
import gym_go
import torch
import numpy as np

# Import the necessary classes and config.
# Ensure that the classes MuZeroAgent, MuZeroNet, and Config are accessible.
# For example, if you saved them in a module named `muzero_model.py`, you could:
# from muzero_model import MuZeroAgent, Config

# (If the definitions are in the same file or available in your PYTHONPATH, adjust accordingly.)

# For this example, we assume the definitions are available in the current context.
# Set device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game configuration.
board_size = 6
env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
action_size = env.action_space.n

# Initialize the agent and load the saved model.
agent = MuZeroAgent(board_size, Config.latent_dim, action_size, num_simulations=50)
agent.net.load_state_dict(torch.load("muzero_model.pth", map_location=device))
agent.net.eval()  # Set to evaluation mode

# Let the human choose their side.
human_side = ""
while human_side not in ["B", "W"]:
    human_side = input("Choose your side (B for Black, W for White): ").strip().upper()

# In Go, Black moves first.
human_turn = True if human_side == "B" else False

print("Starting game. Input your move as two numbers: row col (0-indexed). Type 'pass' to pass.\n")

obs = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # Assumes env.render() displays the board.
    if human_turn:
        valid_input = False
        while not valid_input:
            move = input("Your move (row col) or 'pass': ").strip().lower()
            if move == "pass":
                action = action_size - 1  # Assuming last action is pass.
                valid_input = True
            else:
                try:
                    row, col = map(int, move.split())
                    if 0 <= row < board_size and 0 <= col < board_size:
                        action = row * board_size + col
                        valid_input = True
                    else:
                        print("Coordinates out of range. Please enter values between 0 and", board_size - 1)
                except Exception as e:
                    print("Invalid input. Please enter two numbers separated by a space, or 'pass'.")
    else:
        action, _ = agent.select_action(obs)
        print("Agent plays action:", action)

    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Alternate turn.
    human_turn = not human_turn

env.render()
print("Game over. Total reward:", total_reward)
