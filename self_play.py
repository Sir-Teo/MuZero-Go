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
import time
import argparse
import pickle
from gym_go import govars

warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")

# --- Configuration (copied from play.py, consider moving to a shared config file) ---
class Config:
    board_size = 6
    latent_dim = 96
    mcts_simulations = 128  # Number of simulations per move
    dirichlet_epsilon = 0.02
    dirichlet_alpha = 0.15
    discount = 0.99
    pass_epsilon = 0.01  # Prior weight for pass when board moves exist
    # Self-play specific config
    temperature_moves = 15  # Number of moves to use temperature sampling
    temperature = 1.0      # Initial temperature for sampling actions
    save_interval = 10     # Save game data every N games

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function for adding Dirichlet noise.
def apply_dirichlet_noise(policy, alpha, epsilon):
    dirichlet = np.random.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * dirichlet

# --- Helper Functions for Display and Move Conversion (copied from play.py) ---
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

# --- Network Modules (copied from play.py) ---
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
        value_map = self.value_conv(latent)
        value = value_map.mean(dim=[2, 3])
        value = self.value_fc(value)
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

# --- MCTS Node and MCTS Implementation (modified from play.py) ---
class MCTSNode:
    def __init__(self, latent, prior, terminal=False):
        self.latent = latent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.terminal = terminal
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

class MCTS:
    def __init__(self, muzero_net, action_size, num_simulations, c_puct=2.5):
        self.net = muzero_net
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    def run(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        latent, value, policy_logits = self.net.initial_inference(obs_tensor)
        policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
        invalid_moves = observation[govars.INVD_CHNL]
        valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
        if valid_board.sum() > 0:
            pass_prior = config.pass_epsilon
        else:
            pass_prior = 1.0
        valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
        policy *= valid_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else: # Handle cases where no valid moves exist (rare, but possible)
            policy = np.ones_like(policy) * valid_mask / valid_mask.sum() if valid_mask.sum() > 0 else np.ones_like(policy) / len(policy)

        policy = apply_dirichlet_noise(policy, config.dirichlet_alpha, config.dirichlet_epsilon)
        
        # Ensure policy only contains probabilities for valid moves after noise
        policy *= valid_mask
        policy_sum_after_noise = policy.sum()
        if policy_sum_after_noise > 0:
             policy /= policy_sum_after_noise
        else: # Revert to uniform over valid if noise zeroed everything
             policy = np.ones_like(policy) * valid_mask / valid_mask.sum() if valid_mask.sum() > 0 else np.ones_like(policy) / len(policy)

        root = MCTSNode(latent[0], prior=0)
        for a in range(self.action_size):
            prior = policy[a] if valid_mask[a] > 0 else 0.0
            root.children[a] = {
                'node': None, 'prior': prior, 'visit_count': 0,
                'value_sum': 0, 'action': a
            }
        
        for _ in range(self.num_simulations):
            path, action = self.select_leaf(root, valid_mask)
            leaf = path[-1]
            
            if leaf.terminal:
                # If the leaf is terminal, backpropagate the known value (0 for Go)
                self.backpropagate(path, 0)
                continue

            if action is None: # Should not happen if not terminal, but safety check
                self.backpropagate(path, leaf.value()) # Backpropagate estimated value
                continue
                
            # Expand the selected leaf node
            latent_input = leaf.latent.unsqueeze(0)
            action_tensor = torch.LongTensor([action]).to(device)
            
            with torch.no_grad():
                next_latent, reward, value, policy_logits = self.net.recurrent_inference(latent_input, action_tensor)
            
            policy_child = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
            
            # Apply valid mask to child policy
            # Need observation for the *next* state to get the valid mask, which we don't have easily here.
            # Approximation: use the current valid mask. This might allow exploring invalid moves in the simulation.
            # A better approach would involve predicting the next state's invalid moves or using the env.
            policy_child *= valid_mask 
            policy_child_sum = policy_child.sum()
            if policy_child_sum > 0:
                 policy_child /= policy_child_sum
            else:
                 policy_child = np.ones_like(policy_child) * valid_mask / valid_mask.sum() if valid_mask.sum() > 0 else np.ones_like(policy_child) / len(policy_child)


            child_node = MCTSNode(next_latent[0], prior=0, terminal=False) # Assume not terminal upon expansion
            for a_child in range(self.action_size):
                 prior_child = policy_child[a_child] if valid_mask[a_child] > 0 else 0.0
                 child_node.children[a_child] = {
                     'node': None, 'prior': prior_child, 'visit_count': 0,
                     'value_sum': 0, 'action': a_child
                 }
            
            # The reward from dynamics is for the transition *into* the new state.
            # Value is the predicted value *of* the new state.
            backup_value = reward.item() + config.discount * value.item()
            leaf.children[action]['node'] = child_node
            self.backpropagate(path + [child_node], backup_value)

        # Return root, visit counts (policy target), and root value
        visit_counts = np.array([
            child['visit_count'] for a, child in root.children.items()
        ])
        root_value = root.value()
        return root, visit_counts, root_value

    def select_leaf(self, node, valid_mask):
        path = [node]
        current_action = None # Track the action taken to reach the *next* node
        while node.children:
            # Check if the node is terminal (no valid moves or game rules indicate end)
            # We approximate terminality check here. A proper check needs game state.
            if node.visit_count > 0 and not any(child['prior'] > 0 for a, child in node.children.items()):
                 node.terminal = True # Mark as terminal if explored and no valid children priors
                 return path, None # Reached a terminal leaf state in the search


            # Select the action using PUCT
            q_values = []
            valid_child_actions = []
            
            # Gather Q-values and priors for valid children
            child_stats = {}
            for a, child in node.children.items():
                 if valid_mask[a] > 0 and child['prior'] > 0: # Only consider valid moves with non-zero prior
                     child_node = child['node']
                     q = child_node.value() if (child_node is not None and child_node.visit_count > 0) else 0.0
                     visit_count = child_node.visit_count if child_node is not None else 0
                     child_stats[a] = {'q': q, 'prior': child['prior'], 'visit_count': visit_count}
                     if child_node is not None: # Only consider expanded children for Q-value normalization
                         q_values.append(q)
                         valid_child_actions.append(a)


            # If no valid children found (or all expanded are terminal), treat as leaf
            if not child_stats:
                 node.terminal = True
                 return path, None

            # Normalize Q-values for explored children
            if q_values:
                 min_q = min(q_values)
                 max_q = max(q_values)
            else: # No children have been visited yet
                 min_q, max_q = 0, 0

            best_score = -float('inf')
            best_action = None
            
            # Check unexpanded children first (exploration bonus)
            unexpanded_actions = [a for a, child in node.children.items() if valid_mask[a] > 0 and child['prior'] > 0 and child['node'] is None]
            if unexpanded_actions:
                # Select randomly among unexpanded valid actions
                # This prioritizes exploring new branches
                best_action = random.choice(unexpanded_actions)
                # print(f"Selecting unexpanded action: {best_action}")

            else: # All valid children branches have been visited at least once
                # Calculate PUCT scores only for valid, expanded children
                 for a in valid_child_actions:
                     stats = child_stats[a]
                     q_normalized = (stats['q'] - min_q) / (max_q - min_q) if max_q > min_q else stats['q']
                     
                     # Ensure node.visit_count is at least 1 for the sqrt calculation
                     parent_visit_count_adjusted = max(1, node.visit_count)

                     # Calculate PUCT score
                     u = self.c_puct * stats['prior'] * np.sqrt(parent_visit_count_adjusted) / (1 + stats['visit_count'])
                     score = q_normalized + u

                     # print(f" Action {a}: Q={stats['q']:.2f} (Norm: {q_normalized:.2f}), Prior={stats['prior']:.3f}, N={stats['visit_count']}, U={u:.3f}, Score={score:.3f}")


                     if score > best_score:
                         best_score = score
                         best_action = a
                 # print(f"Selecting best action: {best_action} with score {best_score:.3f}")


            if best_action is None:
                # This might happen if all valid moves have prior 0 or lead to terminal nodes explored
                # Fallback: choose a random valid move if available? Or declare terminal?
                # If no valid action can be selected, treat as terminal leaf
                # print(f"Warning: No best action found at node with visit count {node.visit_count}. Marking terminal.")
                node.terminal = True
                return path, None


            # Move to the selected child node
            selected_child_info = node.children[best_action]
            if selected_child_info['node'] is None:
                 # This indicates we selected an unexpanded node, return path to it
                 return path, best_action
            else:
                 # Move down the tree
                 node = selected_child_info['node']
                 path.append(node)
                 current_action = best_action # Action leading to the *new* node


        # If loop finishes, it means we reached a node with no children dict (shouldn't happen with init)
        # Or reached a node marked terminal previously.
        return path, None

    def backpropagate(self, path, value):
        # Correct backpropagation: value is from the perspective of the player *whose turn it was* at that node.
        # Since MuZero uses a value relative to the current player, we need to negate the value for the opponent's nodes.
        for i, node in enumerate(reversed(path)):
            node.visit_count += 1
            # The value is negated for every step back up the tree, alternating perspectives.
            node.value_sum += value * ((-1) ** i)


# --- MuZero Agent for Self-Play ---
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, env_action_size, num_simulations):
        self.board_size = board_size
        self.action_size = env_action_size

        #max_action_size = int((board_size * board_size * 1.5))
        max_action_size = board_size * board_size +1
        self.net = MuZeroNet(latent_dim, max_action_size).to(device)
        self.net.eval() # Set to evaluation mode
        self.mcts_simulations = num_simulations

    def select_action(self, observation, temperature):
        """
        Selects an action using MCTS, potentially sampling based on temperature.
        Returns the selected action and the normalized visit counts (policy target).
        """
        # Compute valid moves mask from observation
        invalid_moves = observation[govars.INVD_CHNL]
        valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
        # Pass is valid only if no board moves available
        if valid_board.sum() > 0:
            pass_prior = config.pass_epsilon
        else:
            pass_prior = 1.0
        valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
        mcts = MCTS(self.net, self.action_size, self.mcts_simulations)
        with torch.no_grad():
            root, visit_counts, root_value = mcts.run(observation)

        # Mask visit counts to exclude invalid moves
        visit_counts = visit_counts * valid_mask
        # Calculate policy target normalized over valid moves
        policy_target = visit_counts / visit_counts.sum() if visit_counts.sum() > 0 else valid_mask / valid_mask.sum()
        
        # Final action selection: ensure only valid moves
        if temperature == 0:
            # Greedy: choose highest visit count among valid moves
            if visit_counts.sum() > 0:
                action = int(np.argmax(visit_counts))
            else:
                # No visits: choose randomly among valid moves
                valid_actions = np.where(valid_mask > 0)[0]
                action = int(random.choice(valid_actions))
        else:
            # Sample based on visit counts with temperature
            visit_counts_temp = visit_counts**(1.0 / temperature)
            # Mask again to be safe
            visit_counts_temp = visit_counts_temp * valid_mask
            temp_sum = visit_counts_temp.sum()
            if temp_sum > 0:
                probabilities = visit_counts_temp / temp_sum
            else:
                # Uniform distribution over valid moves
                probabilities = valid_mask / valid_mask.sum()
            action = int(np.random.choice(self.action_size, p=probabilities))

        return action, policy_target, root_value

    def load_weights(self, weight_file):
        try:
            state_dict = torch.load(weight_file, map_location=device)
            self.net.load_state_dict(state_dict)
            self.net.eval() # Ensure network is in eval mode after loading
            print(f"Loaded weights from {weight_file}")
        except Exception as e:
            print(f"Error loading weight file '{weight_file}': {e}")
            sys.exit(1)

# --- Game History Storage ---
class GameHistory:
    def __init__(self, board_size, discount):
        self.board_size = board_size
        self.discount = discount
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = [] # Store MCTS values (optional, for training value head)
        self.dones = []
        self.final_reward = 0 # Store the final outcome (-1, 0, 1)

    def add_step(self, obs, action, reward, done, policy, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.policies.append(policy)
        self.values.append(value) # Store root value from MCTS

    def calculate_returns(self):
        """ Calculates discounted returns (G_t) for training the value target. """
        returns = []
        discounted_reward = self.final_reward # Start from the end
        # Iterate backwards from the second-to-last step
        for reward in reversed(self.rewards):
            # The 'reward' stored is the reward received *after* the action.
            # G_t = R_{t+1} + gamma * G_{t+1}
            discounted_reward = reward + self.discount * discounted_reward
            returns.append(discounted_reward)
        
        # Returns are calculated backwards, so reverse them to align with steps
        return list(reversed(returns))

    def __len__(self):
        return len(self.actions)

# --- Self-Play Game Loop ---
def run_self_play_game(agent, env, config):
    """ Runs a single game of self-play. """
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0] # Handle potential tuple return from reset

    done = False
    move_count = 0
    max_moves = config.board_size * config.board_size
    game_history = GameHistory(config.board_size, config.discount)
    start_time = time.time()

    while not done and move_count < max_moves:
        print("\n" + "-"*20)
        current_player = 'Black' if move_count % 2 == 0 else 'White'
        print(f"Move {move_count + 1}, Turn: {current_player}")
        print_board(observation, config.board_size)

        current_temp = config.temperature if move_count < config.temperature_moves else 0
        
        # Get action, policy target, and root value from agent/MCTS
        action, policy, root_value = agent.select_action(observation, current_temp)
        
        print(f"Agent ({current_player}) plays: {action_to_coord(action, config.board_size)} (MCTS Value: {root_value:.4f})")

        # Step environment
        result = env.step(action)
        
        # Unpack result based on Gym version potentially
        if len(result) == 5: # New Gym API (obs, reward, terminated, truncated, info)
            next_observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4: # Older Gym API (obs, reward, done, info)
             next_observation, reward, done, info = result
        else:
             raise ValueError(f"Unexpected number of return values from env.step: {len(result)}")

        if isinstance(next_observation, tuple):
             next_observation = next_observation[0] # Ensure obs is array

        # Store step data (using the actual root_value)
        game_history.add_step(observation, action, reward, done, policy, root_value)

        observation = next_observation
        move_count += 1

        if done:
             # Handle game end: Determine final reward for value target calculation
             # env.winner() gives +1 for black win, -1 for white win, 0 for draw
             winner = env.winner()
             # Ensure the final reward reflects the outcome correctly
             # Typically +1 for win, -1 for loss, 0 for draw from the perspective of the *last* player to move?
             # Or based on the actual winner signal from the env. Let's use env.winner directly.
             game_history.final_reward = winner
             break # Exit loop once game is done

    if not done:
        print(f"Warning: Game reached max moves ({max_moves}) without explicit termination.")
        # Decide how to assign reward in this case. Draw? Or based on score?
        # Using env.winner() might still work if based on score estimate.
        game_history.final_reward = env.winner()


    end_time = time.time()
    game_duration = end_time - start_time
    
    # Print final state
    print("\n" + "-"*20)
    print("Final Board State:")
    print_board(observation, config.board_size)
    winner_map = {1: "Black", -1: "White", 0: "Draw"}
    print(f"Game finished in {move_count} moves ({game_duration:.2f} sec). Winner: {winner_map.get(game_history.final_reward, 'Unknown')}")

    return game_history

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run MuZero self-play.")
    parser.add_argument("--weights", default="/gpfs/scratch/wz1492/MuZero-Go/checkpoints/o0uy97a9/muzero_model_episode_2500.pth", type=str,help="Path to the MuZero model weights (.pth file).")
    parser.add_argument("--num_games", type=int, default=1, help="Number of self-play games to run.")
    parser.add_argument("--output_dir", type=str, default="self_play_data", help="Directory to save game data.")
    parser.add_argument("--simulations", type=int, default=config.mcts_simulations, help="Number of MCTS simulations per move.")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Update config from args if needed
    config.mcts_simulations = args.simulations

    # Initialize environment
    env = gym.make("gym_go:go-v0", size=config.board_size, komi=0, reward_method='real')
    env_action_size = config.board_size * config.board_size + 1

    # Initialize agent
    agent = MuZeroAgent(config.board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    agent.load_weights(args.weights)

    all_game_histories = []
    start_time_total = time.time()

    for i in range(args.num_games):
        print(f"Starting game {i+1}/{args.num_games}...")
        history = run_self_play_game(agent, env, config)
        all_game_histories.append(history)
        print(f"Game {i+1} finished. Moves: {len(history)}, Final Reward: {history.final_reward}")

        # Save periodically
        if (i + 1) % config.save_interval == 0 or (i + 1) == args.num_games:
             data_path = os.path.join(args.output_dir, f"self_play_batch_{i+1}.pkl")
             try:
                 with open(data_path, 'wb') as f:
                     # Save only the necessary parts for training, not full agent/env
                     # Extract data from each history object
                     batch_data = []
                     for h in all_game_histories[i+1-config.save_interval:]: # Save last interval batch
                         returns = h.calculate_returns()
                         batch_data.append({
                             'observations': h.observations,
                             'actions': h.actions,
                             'policies': h.policies,
                             'values': h.values, # MCTS root values
                             'rewards': h.rewards, # Step rewards R_{t+1}
                             'returns': returns, # Discounted returns G_t
                             'final_reward': h.final_reward
                         })

                 pickle.dump(batch_data, f)
                 print(f"Saved batch of {len(batch_data)} games to {data_path}")
             except Exception as e:
                 print(f"Error saving game data: {e}")


    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"Finished {args.num_games} games in {total_duration:.2f} seconds.")
    avg_time = total_duration / args.num_games if args.num_games > 0 else 0
    print(f"Average time per game: {avg_time:.2f} seconds.")

    # Optional: Final save of all data if needed (might be large)
    # final_data_path = os.path.join(args.output_dir, "self_play_all.pkl")
    # with open(final_data_path, 'wb') as f:
    #      pickle.dump(all_game_histories, f)
    # print(f"Saved all {len(all_game_histories)} game histories to {final_data_path}")

if __name__ == "__main__":
    main() 