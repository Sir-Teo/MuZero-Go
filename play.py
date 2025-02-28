#!/usr/bin/env python
import gym
import gym_go
from gym_go import govars
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings

# --- Suppress gym warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# --- Configuration ---
class Config:
    board_size = 9
    latent_dim = 32
    mcts_simulations = 256
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.03

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Network Definitions ---
class RepresentationNetwork(nn.Module):
    def __init__(self, board_size, latent_dim):
        super(RepresentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, board_size, latent_dim, action_size):
        super(DynamicsNetwork, self).__init__()
        self.board_size = board_size
        self.action_embedding = nn.Embedding(action_size, latent_dim)
        self.conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.fc_reward = nn.Linear(latent_dim * board_size * board_size, 1)
        self.relu = nn.ReLU()

    def forward(self, latent, action):
        batch_size = latent.shape[0]
        action_emb = self.action_embedding(action)
        action_emb = action_emb.view(batch_size, latent.shape[1], 1, 1).expand_as(latent)
        x = latent + action_emb
        x = self.relu(self.conv(x))
        reward = self.fc_reward(x.view(batch_size, -1))
        return x, reward

class PredictionNetwork(nn.Module):
    def __init__(self, board_size, latent_dim, action_size):
        super(PredictionNetwork, self).__init__()
        self.board_size = board_size
        self.fc_value = nn.Linear(latent_dim * board_size * board_size, 1)
        self.fc_policy = nn.Linear(latent_dim * board_size * board_size, action_size)

    def forward(self, latent):
        x = latent.view(latent.size(0), -1)
        value = self.fc_value(x)
        policy_logits = self.fc_policy(x)
        return value, policy_logits

class MuZeroNet(nn.Module):
    def __init__(self, board_size, latent_dim, action_size):
        super(MuZeroNet, self).__init__()
        self.board_size = board_size
        self.latent_dim = latent_dim
        self.action_size = action_size
        self.representation = RepresentationNetwork(board_size, latent_dim)
        self.dynamics = DynamicsNetwork(board_size, latent_dim, action_size)
        self.prediction = PredictionNetwork(board_size, latent_dim, action_size)

    def initial_inference(self, observation):
        latent = self.representation(observation)
        value, policy_logits = self.prediction(latent)
        return latent, value, policy_logits

    def recurrent_inference(self, latent, action):
        next_latent, reward = self.dynamics(latent, action)
        value, policy_logits = self.prediction(next_latent)
        return next_latent, reward, value, policy_logits

# --- MCTS with Dirichlet Noise & Invalid Move Masking ---
class MCTSNode:
    def __init__(self, latent, prior):
        self.latent = latent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTS:
    def __init__(self, muzero_net, action_size, num_simulations, c_puct=1.0):
        self.net = muzero_net
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        latent, value, policy_logits = self.net.initial_inference(obs_tensor)
        policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]

        # Get invalid move mask from observation
        invalid_moves = observation[govars.INVD_CHNL]
        valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
        # Append validity for the pass move (always valid)
        valid_mask = np.concatenate([valid_mask, np.array([1.0])])
        if valid_mask[:-1].sum() > 0:
            valid_mask[-1] *= 0.9

        masked_policy = policy * valid_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            masked_policy = valid_mask / valid_mask.sum()

        # Add Dirichlet noise
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(masked_policy))
        masked_policy = (1 - config.dirichlet_epsilon) * masked_policy + config.dirichlet_epsilon * noise

        root = MCTSNode(latent, 0)
        for a in range(self.action_size):
            root.children[a] = {'node': None, 'prior': masked_policy[a],
                                'action': a, 'visit_count': 0, 'value_sum': 0}
        for _ in range(self.num_simulations):
            self.simulate(root)
        return root

    def simulate(self, node):
        best_score = -float('inf')
        best_action = None
        for action, child in node.children.items():
            if child['visit_count'] == 0:
                ucb = self.c_puct * child['prior']
            else:
                ucb = (child['value_sum'] / child['visit_count'] +
                       self.c_puct * child['prior'] * np.sqrt(node.visit_count + 1) / (1 + child['visit_count']))
            if ucb > best_score:
                best_score = ucb
                best_action = action

        selected = node.children[best_action]
        if selected['node'] is None:
            action_tensor = torch.LongTensor([best_action]).to(device)
            next_latent, reward, value, policy_logits = self.net.recurrent_inference(node.latent, action_tensor)
            policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
            child_node = MCTSNode(next_latent, 0)
            for a in range(self.action_size):
                child_node.children[a] = {'node': None, 'prior': policy[a],
                                          'action': a, 'visit_count': 0, 'value_sum': 0}
            selected['node'] = child_node
            selected['visit_count'] += 1
            selected['value_sum'] += value.item()
            return value.item()
        else:
            value_estimate = self.simulate(selected['node'])
            selected['visit_count'] += 1
            selected['value_sum'] += value_estimate
            return value_estimate

# --- MuZero Agent for Inference ---
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, action_size, num_simulations):
        self.board_size = board_size
        self.action_size = action_size
        self.net = MuZeroNet(board_size, latent_dim, action_size).to(device)
        self.mcts_simulations = num_simulations

    def select_action(self, observation):
        # Create a valid move mask
        invalid_moves = observation[govars.INVD_CHNL]
        valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
        valid_mask = np.concatenate([valid_mask, np.array([1.0])])
        if valid_mask[:-1].sum() > 0:
            valid_mask[-1] *= 0.9

        mcts = MCTS(self.net, self.action_size, self.mcts_simulations)
        root = mcts.run(observation)
        visit_counts = np.array([child['visit_count'] if valid_mask[a] > 0 else 0
                                  for a, child in root.children.items()])

        if visit_counts.sum() > 0:
            best_action = int(np.argmax(visit_counts))
        else:
            valid_actions = [a for a, valid in enumerate(valid_mask) if valid > 0]
            best_action = random.choice(valid_actions)
        return best_action, visit_counts

# --- Helper Function: Print Board ---
def print_board(obs, board_size):
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    # Assume govars.BLACK_CHNL and govars.WHITE_CHNL indicate the channels for black and white stones.
    black_channel = obs[govars.BLACK_CHNL]
    white_channel = obs[govars.WHITE_CHNL]
    for i in range(board_size):
        for j in range(board_size):
            if black_channel[i, j] > 0:
                board[i][j] = 'B'
            elif white_channel[i, j] > 0:
                board[i][j] = 'W'
    print("Current board state:")
    for i, row in enumerate(board):
        print(f"{i} " + " ".join(row))
    print("   " + " ".join(str(j) for j in range(board_size)))

# --- Helper Function: Get Human Move ---
def human_move(board_size):
    move = input("Enter your move as 'row,col' (e.g., 3,4) or type 'pass': ")
    if move.lower() == 'pass':
        return board_size * board_size  # Pass move is the last action.
    try:
        row, col = map(int, move.split(','))
        if row < 0 or row >= board_size or col < 0 or col >= board_size:
            print("Move out of bounds. Try again.")
            return human_move(board_size)
        return row * board_size + col
    except Exception:
        print("Invalid input. Please enter in the format 'row,col' or 'pass'.")
        return human_move(board_size)

# --- Main Function for Interactive Play ---
def main():
    board_size = config.board_size
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    action_size = env.action_space.n  # should equal board_size*board_size + 1
    latent_dim = config.latent_dim

    weight_file = input("Enter the path to the model weight file: ").strip()
    agent = MuZeroAgent(board_size, latent_dim, action_size, num_simulations=config.mcts_simulations)
    agent.net.load_state_dict(torch.load(weight_file, map_location=device))
    agent.net.eval()

    color_choice = input("Do you want to play as Black (B) or White (W)? ").strip().upper()
    human_is_black = (color_choice == "B")
    print("You are playing as " + ("Black" if human_is_black else "White"))

    # In Go, Black plays first.
    current_player_is_human = human_is_black  
    done = False

    while not done:
        print_board(obs, board_size)
        if current_player_is_human:
            print("Your turn.")
            action = human_move(board_size)
        else:
            print("Agent's turn. Thinking...")
            action, _ = agent.select_action(obs)
            row = action // board_size
            col = action % board_size
            if action == board_size * board_size:
                print("Agent chose to pass.")
            else:
                print(f"Agent chose move: {row},{col}")
        # Take a step in the environment
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            obs, reward, done, info = result
        if isinstance(obs, tuple):
            obs = obs[0]
        # Switch turns after each move
        current_player_is_human = not current_player_is_human

    print_board(obs, board_size)
    print("Game over!")
    # (Optional) You can add additional logic here to announce the winner.

if __name__ == "__main__":
    main()
