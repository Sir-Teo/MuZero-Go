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
    board_size = 6
    latent_dim = 64
    mcts_simulations = 64
    dirichlet_epsilon = 0.02
    dirichlet_alpha = 0.01
    discount = 0.99
    pass_epsilon = 0.01  # Prior weight for pass when board moves exist

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function for adding Dirichlet noise.
def apply_dirichlet_noise(policy, alpha, epsilon):
    dirichlet = np.random.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * dirichlet

# --- Network Modules ---
class RepresentationNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(RepresentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim, max_action_size):
        super(DynamicsNetwork, self).__init__()
        self.action_embedding = nn.Embedding(max_action_size, latent_dim)
        self.conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.reward_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.fc_reward_hidden = nn.Linear(1, 16)
        self.fc_reward_output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, latent, action):
        batch_size = latent.shape[0]
        action_emb = self.action_embedding(action)
        action_emb = action_emb.view(batch_size, latent.shape[1], 1, 1).expand_as(latent)
        x = latent + action_emb
        x = self.relu(self.conv(x))
        reward_map = self.reward_conv(x)
        reward = reward_map.mean(dim=[2, 3])
        reward = self.relu(self.fc_reward_hidden(reward))
        reward = self.fc_reward_output(reward)
        return x, reward

class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(PredictionNetwork, self).__init__()
        self.value_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.value_fc = nn.Linear(1, 1)
        self.policy_conv = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.pass_logit = nn.Parameter(torch.zeros(1))
    def forward(self, latent):
        # Value head.
        value_map = self.value_conv(latent)
        value = value_map.mean(dim=[2, 3])
        value = self.value_fc(value)
        # Policy head.
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

# --- MCTS Node and MCTS Implementation ---
class MCTSNode:
    def __init__(self, latent, prior, terminal=False):
        self.latent = latent  # hidden state (tensor)
        self.prior = prior    # prior probability for this node
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.terminal = terminal
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTS:
    def __init__(self, muzero_net, action_size, num_simulations, c_puct=10.0):
        self.net = muzero_net
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    def run(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        latent, value, policy_logits = self.net.initial_inference(obs_tensor)
        policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
        # Use the invalid move mask from the observation and penalize pass
        invalid_moves = observation[govars.INVD_CHNL]
        valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
        # Penalize pass when board moves exist, full weight if no board moves left
        if valid_board.sum() > 0:
            pass_prior = config.pass_epsilon
        else:
            pass_prior = 1.0
        valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
        policy *= valid_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        policy = apply_dirichlet_noise(policy, config.dirichlet_alpha, config.dirichlet_epsilon)
        # Initialize root node.
        root = MCTSNode(latent[0], prior=0)
        for a in range(self.action_size):
            prior = policy[a] if valid_mask[a] > 0 else 0
            root.children[a] = {
                'node': None,
                'prior': prior,
                'visit_count': 0,
                'value_sum': 0,
                'action': a
            }
        # Run MCTS simulations.
        for _ in range(self.num_simulations):
            path, action = self.select_leaf(root)
            leaf = path[-1]
            if not leaf.terminal and action is not None:
                latent_input = leaf.latent.unsqueeze(0)
                action_tensor = torch.LongTensor([action]).to(device)
                next_latent, reward, value, policy_logits = self.net.recurrent_inference(latent_input, action_tensor)
                policy_child = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
                child_node = MCTSNode(next_latent[0], prior=0, terminal=False)
                for a in range(self.action_size):
                    prior_child = policy_child[a] if valid_mask[a] > 0 else 0
                    child_node.children[a] = {
                        'node': None,
                        'prior': prior_child,
                        'visit_count': 0,
                        'value_sum': 0,
                        'action': a
                    }
                backup_value = reward.item() + config.discount * value.item()
                leaf.children[action]['node'] = child_node
                self.backpropagate(path + [child_node], backup_value)
        return root
    def select_leaf(self, node):
        path = [node]
        while node.children:
            # If any child is unexpanded, choose one at random.
            unexpanded = [a for a, child in node.children.items() if child['prior'] > 0 and child['node'] is None]
            if unexpanded:
                action = random.choice(unexpanded)
                return path, action
            # Compute Q-values for expanded children.
            q_values = []
            for a, child in node.children.items():
                if child['prior'] <= 0:
                    continue
                child_node = child['node']
                q = child_node.value() if (child_node is not None and child_node.visit_count > 0) else 0.0
                q_values.append(q)
            if q_values:
                min_q = min(q_values)
                max_q = max(q_values)
            else:
                min_q, max_q = 0, 0
            best_score = -float('inf')
            best_action = None
            for a, child in node.children.items():
                if child['prior'] <= 0:
                    continue
                child_node = child['node']
                q = child_node.value() if (child_node is not None and child_node.visit_count > 0) else 0.0
                q_normalized = (q - min_q) / (max_q - min_q) if max_q > min_q else q
                visit_count = child_node.visit_count if (child_node is not None) else 0
                u = self.c_puct * child['prior'] * np.sqrt(node.visit_count + 1) / (1 + visit_count)
                score = q_normalized + u
                if score > best_score:
                    best_score = score
                    best_action = a
            if best_action is None:
                break
            node = node.children[best_action]['node']
            path.append(node)
        return path, None
    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value

# --- MuZero Agent for Inference ---
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, env_action_size, num_simulations):
        self.board_size = board_size
        self.action_size = env_action_size
        max_action_size = int((board_size * board_size * 1.5))
        self.net = MuZeroNet(latent_dim, max_action_size).to(device)
        self.mcts_simulations = num_simulations
    def select_action(self, observation):
        invalid_moves = observation[govars.INVD_CHNL]
        valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
        if valid_board.sum() > 0:
            pass_prior = config.pass_epsilon
        else:
            pass_prior = 1.0
        valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
        mcts = MCTS(self.net, self.action_size, self.mcts_simulations)
        root = mcts.run(observation)
        visit_counts = np.array([
            child['node'].visit_count if (child['node'] is not None and valid_mask[a] > 0) else 0 
            for a, child in root.children.items()
        ])
        if visit_counts.sum() > 0:
            best_action = int(np.argmax(visit_counts))
        else:
            valid_actions = np.nonzero(valid_mask)[0]
            best_action = int(random.choice(valid_actions))
        return best_action

# --- Helper Functions for Display and Move Conversion ---
def print_board(obs, board_size):
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    for i in range(board_size):
        for j in range(board_size):
            if obs[0, i, j] == 1:
                board[i][j] = 'B'
            elif obs[1, i, j] == 1:
                board[i][j] = 'W'
    header = "   " + " ".join(chr(ord('A') + j) for j in range(board_size))
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
    except Exception:
        raise ValueError("Invalid coordinate format")
    col = ord(col_letter) - ord('A')
    return row * board_size + col

# --- Main Interactive Play Loop ---
def main():
    board_size = config.board_size
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    env_action_size = board_size * board_size + 1
    agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    
    # Load weights from a file (provide the file path via command-line or default to "muzero_model_final.pth")
    weight_file = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scratch/wz1492/MuZero-Go/checkpoints/3jxtdof3/muzero_model_episode_500.pth"
    try:
        state_dict = torch.load(weight_file, map_location=device)
        agent.net.load_state_dict(state_dict)
        print(f"Loaded weights from {weight_file}")
    except Exception as e:
        print(f"Error loading weight file: {e}")
        return

    # Choose player color.
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
    # safeguard: prevent infinite interactive games
    move_count = 0
    max_moves = board_size * board_size * 2
    # In Go, Black (first player) starts.
    human_turn = human_is_black
    while not done and move_count < max_moves:
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
        move_count += 1
    # If max moves reached without end, force game over
    if not done:
        print(f"Reached max moves {move_count}, forcing game end.")
        done = True
    print_board(observation, board_size)
    print("Game over!")
    if reward > 0:
        outcome = "win" if human_is_black else "loss"
    elif reward < 0:
        outcome = "loss" if human_is_black else "win"
    else:
        outcome = "draw"
    print(f"Final outcome: {outcome}. Final reward: {reward}")

if __name__ == "__main__":
    main()
