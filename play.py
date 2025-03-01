#!/usr/bin/env python3
import gym
import gym_go
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import sys
from gym_go import govars

warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")

# --- Configuration ---
class Config:
    max_board_size = 9          # Maximum board size used in training (defines fixed action embedding)
    board_size = 6              # Board size for interactive play (can be adjusted)
    latent_dim = 16
    mcts_simulations = 16
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.03

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Network Modules ---
class RepresentationNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(RepresentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim, max_action_size):
        super(DynamicsNetwork, self).__init__()
        self.action_embedding = nn.Embedding(max_action_size, latent_dim)
        self.conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.reward_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.fc_reward = nn.Linear(1, 1)
        self.relu = nn.ReLU()
    def forward(self, latent, action):
        batch_size = latent.shape[0]
        action_emb = self.action_embedding(action)
        action_emb = action_emb.view(batch_size, latent.shape[1], 1, 1).expand_as(latent)
        x = latent + action_emb
        x = self.relu(self.conv(x))
        reward_map = self.reward_conv(x)
        reward = reward_map.mean(dim=[2,3])
        reward = self.fc_reward(reward)
        return x, reward

class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(PredictionNetwork, self).__init__()
        self.value_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.value_fc = nn.Linear(1, 1)
        self.policy_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.pass_logit = nn.Parameter(torch.zeros(1))
    def forward(self, latent):
        # Value head:
        value_map = self.value_conv(latent)
        value = value_map.mean(dim=[2,3])
        value = self.value_fc(value)
        # Policy head:
        policy_map = self.policy_conv(latent)
        batch_size = policy_map.size(0)
        board_policy = policy_map.view(batch_size, -1)
        pass_logit = self.pass_logit.expand(batch_size, 1)
        policy_logits = torch.cat([board_policy, pass_logit], dim=1)
        return value, policy_logits

class MuZeroNet(nn.Module):
    def __init__(self, latent_dim, max_action_size):
        super(MuZeroNet, self).__init__()
        self.representation = RepresentationNetwork(latent_dim)
        self.dynamics = DynamicsNetwork(latent_dim, max_action_size)
        self.prediction = PredictionNetwork(latent_dim)
    def initial_inference(self, observation):
        latent = self.representation(observation)
        value, policy_logits = self.prediction(latent)
        return latent, value, policy_logits
    def recurrent_inference(self, latent, action):
        next_latent, reward = self.dynamics(latent, action)
        value, policy_logits = self.prediction(next_latent)
        return next_latent, reward, value, policy_logits

# --- MCTS for Action Selection ---
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

        # Use invalid move mask from observation:
        invalid_moves = observation[govars.INVD_CHNL]
        valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
        valid_mask = np.concatenate([valid_mask, np.array([1.0])])  # pass is always valid
        if valid_mask[:-1].sum() > 0:
            valid_mask[-1] *= 0.9
        masked_policy = policy * valid_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            masked_policy = valid_mask / valid_mask.sum()
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
    def __init__(self, board_size, latent_dim, env_action_size, num_simulations):
        self.board_size = board_size
        self.action_size = env_action_size  # board_size^2 + 1
        # Fixed max_action_size based on training configuration.
        max_action_size = config.max_board_size * config.max_board_size + 1
        self.net = MuZeroNet(latent_dim, max_action_size).to(device)
        self.mcts_simulations = num_simulations
    def select_action(self, observation):
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
        return best_action

# --- Helper Functions ---
def print_board(obs, board_size):
    # Assumes obs has shape (channels, board_size, board_size)
    # and that channel 0 contains black stones, channel 1 white stones.
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    for i in range(board_size):
        for j in range(board_size):
            if obs[0, i, j] == 1:
                board[i][j] = 'B'
            elif obs[1, i, j] == 1:
                board[i][j] = 'W'
    header = "   " + " ".join(chr(ord('A')+j) for j in range(board_size))
    print(header)
    for i, row in enumerate(board):
        print(f"{i+1:2d} " + " ".join(row))

def action_to_coord(action, board_size):
    if action == board_size * board_size:
        return "pass"
    row = action // board_size
    col = action % board_size
    return f"{chr(ord('A') + col)}{row+1}"

def coord_to_action(coord, board_size):
    if coord.lower() == "pass":
        return board_size * board_size
    coord = coord.strip().upper()
    col_letter = coord[0]
    try:
        row = int(coord[1:]) - 1
    except:
        raise ValueError("Invalid coordinate format")
    col = ord(col_letter) - ord('A')
    return row * board_size + col

# --- Main Interactive Play Loop ---
def main():
    board_size = config.board_size
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    env_action_size = env.action_space.n  # Should be board_size^2 + 1
    agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    
    # Load weight file from command-line argument (or default) 
    weight_file = sys.argv[1] if len(sys.argv) > 1 else "muzero_model_episode_30000.pth"
    try:
        state_dict = torch.load(weight_file, map_location=device)
        agent.net.load_state_dict(state_dict)
        print(f"Loaded weights from {weight_file}")
    except Exception as e:
        print(f"Error loading weight file: {e}")
        return

    # Choose player color
    while True:
        user_color = input("Do you want to play as Black (B) or White (W)? ").strip().upper()
        if user_color in ['B', 'W']:
            break
        print("Invalid input. Please enter 'B' or 'W'.")
    human_is_black = (user_color == 'B')
    
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    done = False
    # In Go, Black (first player) starts.
    human_turn = human_is_black

    while not done:
        print_board(observation, board_size)
        if human_turn:
            move = input("Your move (e.g., A1 or 'pass'): ")
            try:
                action = coord_to_action(move, board_size)
            except Exception as e:
                print("Invalid move format. Try again.")
                continue
        else:
            action = agent.select_action(observation)
            print(f"Agent move: {action_to_coord(action, board_size)}")
        result = env.step(action)
        if len(result) == 5:
            observation, reward, done, truncated, info = result
            done = done or truncated
        else:
            observation, reward, done, info = result
        if isinstance(observation, tuple):
            observation = observation[0]
        human_turn = not human_turn

    print_board(observation, board_size)
    print("Game over!")
    # Determine outcome based on final reward.
    if reward > 0:
        outcome = "win" if human_is_black else "loss"
    elif reward < 0:
        outcome = "loss" if human_is_black else "win"
    else:
        outcome = "draw"
    print(f"Final outcome: {outcome}. Final reward: {reward}")

if __name__ == "__main__":
    main()
