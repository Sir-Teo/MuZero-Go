import gym
import gym_go
from gym_go import govars
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import logging
import wandb
import warnings
import time

# Suppress gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration & Hyperparameters
class Config:
    max_board_size = 6  # Maximum board size (defines action space)
    board_size = 6      # Default training board size
    latent_dim = 16
    learning_rate = 5e-4
    mcts_simulations = 16
    num_episodes = 100000
    num_envs = 64       # Number of parallel environments
    replay_buffer_size = 2000
    batch_size = 64
    unroll_steps = 10
    discount = 0.99
    value_loss_weight = 1.0
    policy_loss_weight = 1.0
    reward_loss_weight = 1.0
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.03
    initial_elo = 1000
    elo_k = 32
    evaluation_interval = 50

config = Config()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network Modules
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

# Vectorized Environment
class VectorGoEnv:
    def __init__(self, num_envs, board_size):
        self.envs = [gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real') 
                     for _ in range(num_envs)]
        self.num_envs = num_envs
        self.dones = [False] * num_envs
        self.last_observations = [None] * num_envs  # Cache last valid observation

    def reset(self):
        observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            observations.append(obs)
            self.dones[i] = False
            self.last_observations[i] = obs  # Initialize with reset observation
        return np.stack(observations)

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not self.dones[i]:
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, done, truncated, info = result
                    done = done or truncated
                else:
                    obs, reward, done, info = result
                if isinstance(obs, tuple):
                    obs = obs[0]
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                self.dones[i] = done
                self.last_observations[i] = obs  # Update last observation
            else:
                # Use the last valid observation for done environments
                observations.append(self.last_observations[i])
                rewards.append(0.0)
                dones.append(True)
                infos.append({'done': True})
        return np.stack(observations), np.array(rewards), np.array(dones), infos
# MCTS Node
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

# Batched MCTS
class BatchedMCTS:
    def __init__(self, muzero_net, num_envs, action_size, num_simulations, c_puct=1.0):
        self.net = muzero_net
        self.num_envs = num_envs
        self.action_size = action_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.trees = [MCTSNode(None, 0) for _ in range(num_envs)]

    def run(self, observations):
        obs_tensor = torch.FloatTensor(observations).to(device)
        latents, values, policy_logits = self.net.initial_inference(obs_tensor)
        policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()

        for i in range(self.num_envs):
            invalid_moves = observations[i, govars.INVD_CHNL]
            valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
            valid_mask = np.concatenate([valid_mask, np.array([1.0])])  # Include pass move
            if valid_mask[:-1].sum() > 0:
                valid_mask[-1] *= 0.9  # Slightly penalize passing

            masked_policy = policy[i] * valid_mask
            if masked_policy.sum() > 0:
                masked_policy /= masked_policy.sum()
            else:
                masked_policy = valid_mask / valid_mask.sum()

            noise = np.random.dirichlet([config.dirichlet_alpha] * len(masked_policy))
            masked_policy = (1 - config.dirichlet_epsilon) * masked_policy + config.dirichlet_epsilon * noise

            root = MCTSNode(latents[i], prior=0)
            for a in range(self.action_size):
                root.children[a] = {
                    'node': None,
                    'prior': masked_policy[a] if valid_mask[a] > 0 else 0,  # Zero prior for invalid moves
                    'action': a,
                    'visit_count': 0,
                    'value_sum': 0
                }
            self.trees[i] = root

        for _ in range(self.num_simulations):
            all_latents, all_actions, expansion_indices = [], [], []
            for env_idx in range(self.num_envs):
                path, action = self.select_leaf(self.trees[env_idx])
                leaf = path[-1]
                if not leaf.terminal and action is not None:
                    all_latents.append(leaf.latent)
                    all_actions.append(action)
                    expansion_indices.append((env_idx, path, action))

            if all_latents:
                latents_tensor = torch.stack(all_latents).to(device)
                actions_tensor = torch.LongTensor(all_actions).to(device)
                next_latents, rewards, values, policy_logits = self.net.recurrent_inference(latents_tensor, actions_tensor)
                policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()

                for idx, (env_idx, path, action) in enumerate(expansion_indices):
                    tree = self.trees[env_idx]
                    leaf = path[-1]
                    next_latent = next_latents[idx]
                    reward = rewards[idx].item()
                    value = values[idx].item()
                    is_terminal = (reward != 0)  # Customize as needed
                    child_node = MCTSNode(next_latent, prior=0, terminal=is_terminal)
                    # Apply valid move mask to child policy
                    next_obs = observations[env_idx]  # This should ideally be the next state; approximate here
                    invalid_moves = next_obs[govars.INVD_CHNL]
                    valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
                    valid_mask = np.concatenate([valid_mask, np.array([1.0])])
                    masked_child_policy = policy[idx] * valid_mask
                    if masked_child_policy.sum() > 0:
                        masked_child_policy /= masked_child_policy.sum()

                    for a in range(self.action_size):
                        child_node.children[a] = {
                            'node': None,
                            'prior': masked_child_policy[a] if valid_mask[a] > 0 else 0,
                            'action': a,
                            'visit_count': 0,
                            'value_sum': 0
                        }
                    leaf.children[action]['node'] = child_node
                    self.backpropagate(path + [child_node], value)

        return self.trees

    def select_leaf(self, tree):
        node = tree
        path = [node]
        while node.children and all(child['node'] is not None for child in node.children.values() if child['prior'] > 0):
            best_score = -float('inf')
            best_action = None
            for action, child in node.children.items():
                if child['prior'] > 0:  # Only consider valid moves
                    Q = child['value_sum'] / child['visit_count'] if child['visit_count'] > 0 else 0
                    U = self.c_puct * child['prior'] * np.sqrt(node.visit_count + 1) / (1 + child['visit_count'])
                    score = Q + U
                    if score > best_score:
                        best_score = score
                        best_action = action
            if best_action is None:
                break
            node = node.children[best_action]['node']
            path.append(node)
        if node.terminal or not node.children:
            action = None
        else:
            unexpanded = [a for a, child in node.children.items() if child['node'] is None and child['prior'] > 0]
            action = random.choice(unexpanded) if unexpanded else None
        return path, action

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value

# MuZero Agent
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, env_action_size, num_simulations):
        self.board_size = board_size
        self.action_size = env_action_size
        max_action_size = config.max_board_size * config.max_board_size + 1
        self.net = MuZeroNet(latent_dim, max_action_size).to(device)
        self.mcts_simulations = num_simulations
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        loss_total = 0
        self.optimizer.zero_grad()
        batch = random.sample(replay_buffer, batch_size)
        scaler = torch.amp.GradScaler('cuda')  # Updated API
        losses = 0
        with torch.amp.autocast('cuda'):  # Updated API
            for trajectory in batch:
                traj_length = len(trajectory['actions'])
                start_index = random.randint(0, traj_length)
                initial_obs = trajectory['observations'][start_index]
                initial_obs_tensor = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
                latent, value, policy_logits = self.net.initial_inference(initial_obs_tensor)

                step_losses = 0
                target_value = self.compute_target_value(trajectory, start_index, config.unroll_steps)
                target_policy = trajectory['policies'][start_index] if start_index < len(trajectory['policies']) else np.ones(self.action_size) / self.action_size
                target_value_array = np.array(target_value, dtype=np.float32)
                target_policy_array = np.array(target_policy, dtype=np.float32)

                target_value_tensor = torch.tensor(target_value_array, dtype=torch.float32, device=device)  # Scalar
                target_policy_tensor = torch.tensor(target_policy_array, dtype=torch.float32, device=device)  # No list wrapping

                value_loss = F.mse_loss(value.squeeze(), target_value_tensor)
                policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=1), target_policy_tensor, reduction='batchmean')
                step_losses += (config.value_loss_weight * value_loss +
                                config.policy_loss_weight * policy_loss)

                current_latent = latent
                for k in range(1, config.unroll_steps + 1):
                    if start_index + k - 1 < len(trajectory['actions']):
                        action = trajectory['actions'][start_index + k - 1]
                        action_tensor = torch.LongTensor([action]).to(device)
                    else:
                        action_tensor = torch.LongTensor([self.action_size - 1]).to(device)

                    current_latent, reward, value, policy_logits = self.net.recurrent_inference(current_latent, action_tensor)

                    target_reward = trajectory['rewards'][start_index + k - 1] if start_index + k - 1 < len(trajectory['rewards']) else 0.0
                    target_policy = trajectory['policies'][start_index + k] if start_index + k < len(trajectory['policies']) else np.ones(self.action_size) / self.action_size
                    target_value = self.compute_target_value(trajectory, start_index + k, config.unroll_steps)

                    target_reward_tensor = torch.tensor(target_reward, dtype=torch.float32, device=device)  # Scalar
                    target_value_tensor = torch.tensor(target_value, dtype=torch.float32, device=device)  # Scalar
                    target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32, device=device)  # No list wrapping

                    reward_loss = F.mse_loss(reward.squeeze(), target_reward_tensor)
                    value_loss = F.mse_loss(value.squeeze(), target_value_tensor)
                    policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=1), target_policy_tensor, reduction='batchmean')
                    step_losses += (config.reward_loss_weight * reward_loss +
                                    config.value_loss_weight * value_loss +
                                    config.policy_loss_weight * policy_loss)
                losses += step_losses
                loss_total += step_losses.item()
        scaler.scale(losses).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return loss_total / batch_size

    def compute_target_value(self, trajectory, index, unroll_steps):
        target = 0.0
        discount_factor = 1.0
        T = len(trajectory['rewards'])
        for k in range(unroll_steps):
            j = index + k
            if j < T:
                target += discount_factor * trajectory['rewards'][j]
                discount_factor *= config.discount
            else:
                break
        if index + unroll_steps < len(trajectory['observations']):
            obs = trajectory['observations'][index + unroll_steps]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value, _ = self.net.initial_inference(obs_tensor)
            target += discount_factor * value.item()
        return target

# Evaluator
class Evaluator:
    def __init__(self, agent, env, num_eval_episodes=5):
        self.agent = agent
        self.env = env
        self.num_eval_episodes = num_eval_episodes
        self.episodes = []
        self.avg_rewards = []
        self.win_rates = []
        self.elo_ratings = []
        self.elo_rating = config.initial_elo

    def evaluate(self, episode):
        total_rewards = []
        wins = 0
        self.agent.net.eval()
        with torch.no_grad():
            for _ in range(self.num_eval_episodes):
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                done = False
                total_reward = 0
                while not done:
                    mcts = BatchedMCTS(self.agent.net, 1, self.agent.action_size, self.agent.mcts_simulations)
                    roots = mcts.run(np.expand_dims(obs, 0))
                    root = roots[0]
                    visit_counts = np.array([child['visit_count'] for child in root.children.values()])
                    action = int(np.argmax(visit_counts)) if visit_counts.sum() > 0 else random.choice(range(self.agent.action_size))
                    result = self.env.step(action)
                    if len(result) == 5:
                        obs, reward, done, truncated, info = result
                        done = done or truncated
                    else:
                        obs, reward, done, info = result
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    total_reward += reward
                total_rewards.append(total_reward)
                if total_reward > 0:
                    wins += 1
        self.agent.net.train()
        avg_reward = np.mean(total_rewards)
        win_rate = wins / self.num_eval_episodes

        opponent_rating = 1000
        expected = 1 / (1 + 10 ** ((opponent_rating - self.elo_rating) / 400))
        self.elo_rating = self.elo_rating + config.elo_k * (win_rate - expected)

        self.episodes.append(episode)
        self.avg_rewards.append(avg_reward)
        self.win_rates.append(win_rate)
        self.elo_ratings.append(self.elo_rating)
        return avg_reward, win_rate, self.elo_rating

# Main Training Loop
def main():
    board_size = config.board_size
    env_action_size = board_size * board_size + 1
    agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    vec_env = VectorGoEnv(config.num_envs, board_size)
    batched_mcts = BatchedMCTS(agent.net, config.num_envs, env_action_size, config.mcts_simulations)
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    evaluator = Evaluator(agent, gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real'), num_eval_episodes=5)
    
    episode_count = 0
    start_time = time.time()  # To track elapsed time

    while episode_count < config.num_episodes:
        observations = vec_env.reset()
        done = [False] * config.num_envs
        trajectories = [{'observations': [obs], 'actions': [], 'rewards': [], 'policies': []} 
                        for obs in observations]
        new_episodes = 0  # Track new episodes in this batch

        while not all(done):
            roots = batched_mcts.run(observations)
            actions, policies = [], []
            for i, root in enumerate(roots):
                if not done[i]:
                    invalid_moves = observations[i, govars.INVD_CHNL]
                    valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
                    valid_mask = np.concatenate([valid_mask, np.array([1.0])])
                    visit_counts = np.array([child['visit_count'] if valid_mask[a] > 0 else 0 
                                             for a, child in root.children.items()])
                    if visit_counts.sum() > 0:
                        action = int(np.argmax(visit_counts))
                        policy = visit_counts / visit_counts.sum()
                    else:
                        valid_actions = [a for a, v in enumerate(valid_mask) if v > 0]
                        action = random.choice(valid_actions)
                        policy = np.zeros_like(visit_counts)
                        policy[action] = 1.0
                    actions.append(action)
                    policies.append(policy)
                else:
                    actions.append(env_action_size - 1)  # Pass move
                    policies.append(np.zeros(env_action_size))

            next_observations, rewards, dones, infos = vec_env.step(actions)
            for i in range(config.num_envs):
                if not done[i]:
                    trajectories[i]['actions'].append(actions[i])
                    trajectories[i]['rewards'].append(rewards[i])
                    trajectories[i]['policies'].append(policies[i])
                    trajectories[i]['observations'].append(next_observations[i])
                    if dones[i]:
                        done[i] = True
                        replay_buffer.append(trajectories[i])
                        episode_count += 1
                        new_episodes += 1
                        elapsed = time.time() - start_time
                        logger.info(f"Episode {episode_count} completed | Total Episodes: {episode_count} | Elapsed Time: {elapsed:.2f}s")
                        # Log to wandb as well if needed:
                        wandb.log({"episode_complete": episode_count, "elapsed_time": elapsed})

            observations = next_observations

        # Training updates for the batch of new episodes
        batch_losses = []
        for step in range(new_episodes):
            loss = agent.train(replay_buffer, config.batch_size)
            if loss is not None:
                current_episode = episode_count - new_episodes + step + 1
                batch_losses.append(loss)
                logger.info(f"Training Update | Episode {current_episode} | Loss: {loss:.4f}")
                wandb.log({"training_loss": loss, "episode": current_episode})
        if batch_losses:
            avg_loss = np.mean(batch_losses)
            logger.info(f"Batch Training Summary | Episodes: {new_episodes} | Average Loss: {avg_loss:.4f}")

        # Evaluation at fixed intervals
        if episode_count > 0 and episode_count % config.evaluation_interval == 0:
            avg_eval_reward, win_rate, current_elo = evaluator.evaluate(episode_count)
            logger.info(f"Evaluation at Episode {episode_count} | Avg Reward: {avg_eval_reward:.2f} | Win Rate: {win_rate:.2f} | Elo Rating: {current_elo:.2f}")
            wandb.log({
                "evaluation_avg_reward": avg_eval_reward,
                "evaluation_win_rate": win_rate,
                "elo_rating": current_elo,
                "episode": episode_count
            })
            if episode_count % 10000 == 0:
                model_path = f"muzero_model_episode_{episode_count}.pth"
                torch.save(agent.net.state_dict(), model_path)
                wandb.save(model_path)

    # Final model save and finish
    torch.save(agent.net.state_dict(), "muzero_model_final.pth")
    wandb.save("muzero_model_final.pth")
    wandb.finish()

if __name__ == "__main__":
    wandb.init(project="muzero_go_vectorized", config=vars(config))
    logger.info("Starting training process...")
    main()
    logger.info("Training completed.")