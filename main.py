import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import logging
import wandb
import warnings
import time
import gym
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR

from gym_go import govars, rendering, gogame

# Suppress gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration & Hyperparameters
class Config:
    board_size = 6      # Fixed board size (used for training and maximum action space)
    latent_dim = 128
    max_action_size = board_size * board_size + 1 # Explicit action size for network
    learning_rate = 1e-4
    mcts_simulations = 256      # increased MCTS rollouts for stronger self-play targets
    num_episodes = 50000
    replay_buffer_size = 5000   # larger buffer for more diverse experience
    batch_size = 128            # larger batches for stable updates
    unroll_steps = 10
    discount = 0.99
    value_loss_weight = 1.0
    policy_loss_weight = 1.0
    reward_loss_weight = 1.0
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.03
    initial_elo = 1000
    elo_k = 32
    evaluation_interval = 10
    evaluation_num_games = 50   # Number of games to play during evaluation match
    evaluation_win_threshold = 0.55  # Win rate threshold to promote new weights
    # New flags for reward/value scaling and prioritized replay
    use_value_transform = True     # Set True for visually complex domains (e.g. Atari)
    use_prioritized_replay = True  # Enable prioritized replay for faster, focused learning
    prioritized_replay_beta = 0.5  # Exponent for importance sampling weights
    pass_epsilon = 0.05  # Prior weight for pass when board moves exist
    max_game_moves = int((board_size * board_size * 1.5)) 
    num_replay_versions = 1   # number of past model versions to maintain replay buffers for

config = Config()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function: Dirichlet noise is added to the root policy for exploration.
def apply_dirichlet_noise(policy, alpha, epsilon):
    dirichlet = np.random.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * dirichlet

# Reward/Value scaling transform as suggested in the paper for Atari.
def reward_value_transform(x, epsilon=0.001):
    # h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x

# Network Modules
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

# MCTS Node for a single environment.
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

# Prioritized Replay Buffer implementation (simple version)
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
    def add(self, trajectory):
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
            self.priorities.append(1.0)
        else:
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(trajectory)
            self.priorities.append(1.0)
    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices
    def update_priorities(self, indices, new_priorities):
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p

# Multi-version replay buffer to maintain buffers across past model versions
class MultiVersionReplayBuffer:
    def __init__(self, capacity, num_versions, prioritized):
        self.capacity = capacity
        self.num_versions = num_versions
        self.prioritized = prioritized
        self.buffers = deque(maxlen=num_versions)
        self.add_version()

    def add_version(self):
        if self.prioritized:
            buf = PrioritizedReplayBuffer(self.capacity)
        else:
            buf = deque(maxlen=self.capacity)
        self.buffers.append(buf)

    def add(self, trajectory):
        if self.prioritized:
            self.buffers[-1].add(trajectory)
        else:
            self.buffers[-1].append(trajectory)

    def sample(self, batch_size):
        if self.prioritized:
            combined = []
            combined_p = []
            for buf in self.buffers:
                combined.extend(buf.buffer)
                combined_p.extend(buf.priorities)
            if len(combined) < batch_size:
                return [], []
            probs = np.array(combined_p) / sum(combined_p)
            indices = np.random.choice(len(combined), batch_size, replace=False, p=probs)
            samples = [combined[i] for i in indices]
            boundaries = []
            cum = 0
            for buf in self.buffers:
                boundaries.append(cum)
                cum += len(buf.buffer)
            boundaries.append(cum)
            mapping = []
            for i in indices:
                for idx_buf in range(len(boundaries)-1):
                    if boundaries[idx_buf] <= i < boundaries[idx_buf+1]:
                        local = i - boundaries[idx_buf]
                        mapping.append((idx_buf, local))
                        break
            return samples, mapping
        else:
            combined = []
            for buf in self.buffers:
                combined.extend(buf)
            if len(combined) < batch_size:
                return [], []
            samples = random.sample(combined, batch_size)
            return samples, None

    def update_priorities(self, mapping, new_priorities):
        if not self.prioritized or mapping is None:
            return
        for (buf_idx, local_idx), p in zip(mapping, new_priorities):
            if buf_idx < len(self.buffers):
                self.buffers[buf_idx].priorities[local_idx] = p

# Monte Carlo Tree Search (MCTS) implementation with Q-value normalization.
class MCTS:
    def __init__(self, muzero_net, action_size, num_simulations, c_puct=2):
        self.net = muzero_net
        self.net.eval()
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    @torch.no_grad()
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

        # Zero-out invalid moves and normalize.
        policy *= valid_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum

        # Apply Dirichlet noise at the root.
        policy = apply_dirichlet_noise(policy, config.dirichlet_alpha, config.dirichlet_epsilon)

        # Initialize root node.
        root = MCTSNode(latent[0], prior=0)
        for a in range(self.action_size):
            prior = policy[a] if valid_mask[a] > 0 else 0
            root.children[a] = {
                'node': None,
                'prior': prior,
                'visit_count': 0,  # Not updated separately here.
                'value_sum': 0,
                'action': a
            }

        total_visits = 0
        for _ in range(self.num_simulations):
            path, action = self.select_leaf(root)
            leaf = path[-1]
            if not leaf.terminal and action is not None:
                latent_input = leaf.latent.unsqueeze(0)
                action_tensor = torch.LongTensor([action]).to(device)
                next_latent, reward, value, policy_logits = self.net.recurrent_inference(latent_input, action_tensor)
                policy_child = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]

                # Expand the leaf node.
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
                total_visits += 1

        return root

    def select_leaf(self, node):
        path = [node]
        while node.children:
            # If any child is unexpanded, choose one at random.
            unexpanded = [a for a, child in node.children.items() if child['prior'] > 0 and child['node'] is None]
            if unexpanded:
                action = random.choice(unexpanded)
                return path, action

            # Compute raw Q-values for expanded children.
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
                # Normalize Q value using min-max normalization.
                if max_q > min_q:
                    q_normalized = (q - min_q) / (max_q - min_q)
                else:
                    q_normalized = q
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

# MuZero Agent that uses the network and MCTS for self-play and training.
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, env_action_size, num_simulations):
        self.board_size = board_size
        self.action_size = env_action_size
        self.net = MuZeroNet(latent_dim, config.max_action_size).to(device)
        self.mcts_simulations = num_simulations
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        # set up a learning-rate scheduler for gradual annealing
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def train(self, replay_buffer, batch_size):
        # Sample a batch from the (possibly multi-version) replay buffer.
        batch, indices = replay_buffer.sample(batch_size)
        if not batch or len(batch) < batch_size:
            return None
        loss_total = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_reward_loss = 0.0

        self.optimizer.zero_grad()
        for trajectory in batch:
            traj_length = len(trajectory['actions'])
            start_index = random.randint(0, traj_length - 1)
            initial_obs = trajectory['observations'][start_index]
            initial_obs_tensor = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
            latent, value, policy_logits = self.net.initial_inference(initial_obs_tensor)

            step_loss = 0.0
            # Compute targets for the initial inference.
            target_value = self.compute_target_value(trajectory, start_index, config.unroll_steps)
            target_policy = (
                trajectory['policies'][start_index]
                if start_index < len(trajectory['policies'])
                else np.ones(self.action_size) / self.action_size
            )
            target_value_tensor = torch.tensor(target_value, dtype=torch.float32, device=device)
            target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32, device=device)

            # Initial inference losses (no reward loss here).
            if config.use_value_transform:
                v_loss = F.mse_loss(
                    reward_value_transform(value.squeeze()),
                    reward_value_transform(target_value_tensor)
                )
            else:
                v_loss = F.mse_loss(value.squeeze(), target_value_tensor)
            p_loss = F.kl_div(
                F.log_softmax(policy_logits, dim=1),
                target_policy_tensor,
                reduction='batchmean'
            )
            current_loss = config.value_loss_weight * v_loss + config.policy_loss_weight * p_loss
            step_loss += current_loss

            total_value_loss += config.value_loss_weight * v_loss.item()
            total_policy_loss += config.policy_loss_weight * p_loss.item()

            current_latent = latent
            # Unroll steps losses.
            for k in range(1, config.unroll_steps + 1):
                if start_index + k - 1 < len(trajectory['actions']):
                    action = trajectory['actions'][start_index + k - 1]
                    action_tensor = torch.LongTensor([action]).to(device)
                else:
                    # Use pass action if we've run out of recorded actions.
                    action_tensor = torch.LongTensor([self.action_size - 1]).to(device)

                current_latent, reward, value, policy_logits = self.net.recurrent_inference(current_latent, action_tensor)

                target_reward = (
                    trajectory['rewards'][start_index + k - 1]
                    if start_index + k - 1 < len(trajectory['rewards'])
                    else 0.0
                )
                target_policy = (
                    trajectory['policies'][start_index + k]
                    if start_index + k < len(trajectory['policies'])
                    else np.ones(self.action_size) / self.action_size
                )
                target_value = self.compute_target_value(trajectory, start_index + k, config.unroll_steps)

                target_reward_tensor = torch.tensor(target_reward, dtype=torch.float32, device=device)
                target_value_tensor = torch.tensor(target_value, dtype=torch.float32, device=device)
                target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32, device=device)

                r_loss = F.mse_loss(reward.squeeze(), target_reward_tensor)
                if config.use_value_transform:
                    v_loss = F.mse_loss(
                        reward_value_transform(value.squeeze()),
                        reward_value_transform(target_value_tensor)
                    )
                else:
                    v_loss = F.mse_loss(value.squeeze(), target_value_tensor)
                p_loss = F.kl_div(
                    F.log_softmax(policy_logits, dim=1),
                    target_policy_tensor,
                    reduction='batchmean'
                )

                current_loss = (config.reward_loss_weight * r_loss +
                                config.value_loss_weight * v_loss +
                                config.policy_loss_weight * p_loss)
                step_loss += current_loss

                total_reward_loss += config.reward_loss_weight * r_loss.item()
                total_value_loss += config.value_loss_weight * v_loss.item()
                total_policy_loss += config.policy_loss_weight * p_loss.item()

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_total += step_loss.item()

        avg_loss = loss_total / batch_size

        # Log the average loss and individual loss components.
        wandb.log({
            "training_loss": avg_loss,
            "value_loss": total_value_loss / batch_size,
            "policy_loss": total_policy_loss / batch_size,
            "reward_loss": total_reward_loss / batch_size
        })

        if indices is not None:
            new_priorities = [avg_loss] * len(indices)
            replay_buffer.update_priorities(indices, new_priorities)
        # step the learning-rate scheduler and log the current lr
        self.scheduler.step()
        wandb.log({"lr": self.optimizer.param_groups[0]["lr"]})
        return avg_loss

    def compute_target_value(self, trajectory, index, unroll_steps):
        target = 0.0
        discount_factor = 1.0
        T = len(trajectory['rewards'])
        # sum up all actual rewards
        for k in range(unroll_steps):
            j = index + k
            if j < T:
                target += discount_factor * trajectory['rewards'][j]
                discount_factor *= config.discount
            else:
                break
        # only bootstrap if we haven't reached the end of the game
        if index + unroll_steps < T:
            obs = trajectory['observations'][index + unroll_steps]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value, _ = self.net.initial_inference(obs_tensor)
            target += discount_factor * value.item()
        return target


# Self-play evaluator to compare current agent against best agent.
class SelfPlayEvaluator:
    def __init__(self, current_agent, best_agent, env, num_games=20):
        self.current_agent = current_agent
        self.best_agent = best_agent
        self.env = env
        self.num_games = num_games
        self.current_elo = config.initial_elo
        self.best_elo = config.initial_elo

    def play_game(self, starting_player=0):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        turn = starting_player
        move_count = 0
        max_moves = config.max_game_moves
        pass_count = 0  # track consecutive passes
        pass_action = self.current_agent.action_size - 1  # index for pass action
        # play until game ends or max moves reached
        while not done and move_count < max_moves:
            if turn == 0:
                mcts = MCTS(self.current_agent.net, self.current_agent.action_size, self.current_agent.mcts_simulations)
            else:
                mcts = MCTS(self.best_agent.net, self.best_agent.action_size, self.best_agent.mcts_simulations)
            root = mcts.run(obs)
            invalid_moves = obs[govars.INVD_CHNL]
            valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
            if valid_board.sum() > 0:
                pass_prior = config.pass_epsilon
            else:
                pass_prior = 1.0
            valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
            visit_counts = np.array([
                child['node'].visit_count if (child['node'] is not None and valid_mask[a] > 0) else 0 
                for a, child in root.children.items()
            ])
            if visit_counts.sum() > 0:
                action = int(np.argmax(visit_counts))
            else:
                valid_actions = np.nonzero(valid_mask)[0]
                action = int(random.choice(valid_actions))
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
                done = done or truncated
            else:
                obs, reward, done, info = result
            if isinstance(obs, tuple):
                obs = obs[0]
            # increment move counter and toggle turn
            move_count += 1
            turn = 1 - turn
            # Update consecutive pass count to detect game end
            if action == pass_action:
                pass_count += 1
            else:
                pass_count = 0
            if pass_count >= 2:
                done = True
                break
        # if we exited due to move limit, log warning and force end
        if move_count >= max_moves:
            logger.warning(f"Evaluation game reached max moves ({move_count}), forcing end.")
            done = True
        winner = self.env.winner()
        if starting_player == 0:
            return 1 if winner == 1 else 0
        else:
            return 1 if winner == -1 else 0

    def evaluate(self):
        current_wins = 0
        for i in range(self.num_games):
            starting_player = i % 2
            result = self.play_game(starting_player=starting_player)
            if starting_player == 1:
                result = 1 - result
            current_wins += result
        win_rate = current_wins / self.num_games
        expected_score = 1 / (1 + 10 ** ((self.best_elo - self.current_elo) / 400))
        if win_rate > config.evaluation_win_threshold:
            self.current_elo += config.elo_k * (win_rate - expected_score)
            self.best_elo += config.elo_k * ((1 - win_rate) - (1 - expected_score))
        wandb.log({"selfplay_win_rate": win_rate, "current_elo": self.current_elo})
        return win_rate, self.current_elo

# Main training loop using a single environment.
def main():
    # Create a unique checkpoint directory for this run
    base_ckpt_dir = "checkpoints"
    os.makedirs(base_ckpt_dir, exist_ok=True)
    # Use WandB run ID if available, else timestamp
    run_id = wandb.run.id if (hasattr(wandb, 'run') and wandb.run is not None) else time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_ckpt_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    board_size = config.board_size  # Fixed board size of 6
    env_action_size = config.max_action_size
    agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    best_agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    best_agent.net.load_state_dict(agent.net.state_dict())
    
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    # Initialize multi-version replay buffer
    replay_buffer = MultiVersionReplayBuffer(config.replay_buffer_size, config.num_replay_versions, config.use_prioritized_replay)
    selfplay_evaluator = SelfPlayEvaluator(agent, best_agent, env, num_games=config.evaluation_num_games)
    
    episode_count = 0
    last_time = time.time()

    while episode_count < config.num_episodes:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'policies': []
        }
        done = False
        episode_reward = 0.0  # Total reward per episode
        move_rewards = []
        move_max_visits = []
        move_count = 0
        max_moves = config.max_game_moves  # safeguard against infinite self-play
        pass_count = 0  # track consecutive passes
        pass_action = env_action_size - 1  # index for pass action

        # Self-play game (episode)
        while not done and move_count < max_moves:
            mcts = MCTS(agent.net, env_action_size, config.mcts_simulations)
            root = mcts.run(obs)
            invalid_moves = obs[govars.INVD_CHNL]
            valid_board = (invalid_moves.flatten() == 0).astype(np.float32)
            if valid_board.sum() > 0:
                pass_prior = config.pass_epsilon
            else:
                pass_prior = 1.0
            valid_mask = np.concatenate([valid_board, np.array([pass_prior])])
            visit_counts = np.array([
                child['node'].visit_count if (child['node'] is not None and valid_mask[a] > 0) else 0 
                for a, child in root.children.items()
            ])
            if visit_counts.sum() > 0:
                action = int(np.argmax(visit_counts))
                policy = visit_counts / visit_counts.sum()
            else:
                valid_actions = np.nonzero(valid_mask)[0]
                action = int(random.choice(valid_actions))
                policy = np.zeros(env_action_size)
                policy[action] = 1.0

            trajectory['actions'].append(action)
            trajectory['policies'].append(policy)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
                done = done or truncated
            else:
                obs, reward, done, info = result
            if isinstance(obs, tuple):
                obs = obs[0]
            trajectory['rewards'].append(reward)
            trajectory['observations'].append(obs)
            episode_reward += reward
            move_rewards.append(reward)
            move_max_visits.append(int(np.max(visit_counts)))
            move_count += 1
            # Update consecutive pass count to detect game end
            if action == pass_action:
                pass_count += 1
            else:
                pass_count = 0
            if pass_count >= 2:
                done = True
                break
        if not done:
            logger.warning(f"Episode {episode_count+1}: reached max moves {move_count}, forcing end.")
            done = True
            trajectory['rewards'][-1] += -0.5  

        # --- Append final game outcome reward so trajectories carry win/loss signal ---
        winner = env.winner()           # +1 if black wins, -1 if white wins
        trajectory['rewards'].append(float(winner))
        trajectory['observations'].append(obs)  # terminal state
        # ---------------------------------------------------------------------------
        replay_buffer.add(trajectory)

        episode_count += 1
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time
        avg_move_reward = np.mean(move_rewards) if move_rewards else 0.0
        avg_move_max_visit = np.mean(move_max_visits) if move_max_visits else 0.0

        wandb.log({
            "episode_complete": episode_count,
            "elapsed_time": elapsed,
            "episode_reward": episode_reward,
            "avg_move_reward": avg_move_reward,
            "avg_move_max_visit": avg_move_max_visit
        })
        logger.info(f"Episode {episode_count} completed | Elapsed Time: {elapsed:.2f}s | Episode Reward: {episode_reward:.2f}")

        # Training update after each episode.
        loss = agent.train(replay_buffer, config.batch_size)
        if loss is not None:
            logger.info(f"Training Update | Episode {episode_count} | Loss: {loss:.4f}")

        # Self-play evaluation at fixed intervals.
        if episode_count > 0 and episode_count % config.evaluation_interval == 0:
            win_rate, new_elo = selfplay_evaluator.evaluate()
            logger.info(f"Self-play Evaluation at Episode {episode_count} | Win Rate: {win_rate:.2f} | Elo: {new_elo:.2f}")
            wandb.log({"selfplay_win_rate": win_rate, "selfplay_elo": new_elo, "episode": episode_count})
            if win_rate > config.evaluation_win_threshold:
                best_agent.net.load_state_dict(agent.net.state_dict())
                logger.info(f"New agent promoted as best with win rate {win_rate:.2f}.")
                # Start collecting self-play data in a new buffer for the promoted model version
                replay_buffer.add_version()
            else:
                agent.net.load_state_dict(best_agent.net.state_dict())
                logger.info(f"Reverted agent to previous best due to insufficient win rate {win_rate:.2f}.")

            if episode_count % 100 == 0:
                # Save checkpoint every 100 episodes
                model_path = os.path.join(checkpoint_dir, f"muzero_model_episode_{episode_count}.pth")
                torch.save(agent.net.state_dict(), model_path)
                wandb.save(model_path)
                logger.info(f"Saved checkpoint weights at episode {episode_count} to {model_path}")
            # reset timer after evaluation to avoid including evaluation overhead in next episode's elapsed time
            last_time = time.time()

    # Save final model weights
    final_model_path = os.path.join(checkpoint_dir, "muzero_model_final.pth")
    torch.save(agent.net.state_dict(), final_model_path)
    wandb.save(final_model_path)
    logger.info(f"Saved final model weights to {final_model_path}")
    wandb.finish()

if __name__ == "__main__":
    wandb.init(project="muzero_go_0427", config=vars(config))
    logger.info("Starting training process...")
    main()
    logger.info("Training completed.")
