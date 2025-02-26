import gym
import gym_go
from gym_go import govars  # Import govars to access channel indices
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
from multiprocessing import Pool
import torch.multiprocessing as mp
import argparse
import os
import pickle

# --- Suppress gym warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
class Config:
    board_size = 9
    latent_dim = 64
    learning_rate = 5e-4
    mcts_simulations = 64
    num_episodes = 500000
    replay_buffer_size = 10000
    batch_size = 128
    unroll_steps = 10
    discount = 0.99
    value_loss_weight = 1.0
    policy_loss_weight = 1.0
    reward_loss_weight = 1.0
    # New parameters for exploration noise:
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.03
    # ELO update parameters:
    initial_elo = 1000
    elo_k = 32
    evaluation_interval = 50

config = Config()

# --- Set device to GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Network Modules ---
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

# --- MCTS with Invalid Move Masking & Dirichlet Noise ---
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

        # Extract invalid move mask from observation
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

        # Add Dirichlet noise to the root prior
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

# --- MuZero Agent with Fixed Target Value Calculation ---
class MuZeroAgent:
    def __init__(self, board_size, latent_dim, action_size, num_simulations):
        self.board_size = board_size
        self.action_size = action_size
        self.net = MuZeroNet(board_size, latent_dim, action_size).to(device)
        self.mcts_simulations = num_simulations
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)

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

        if visit_counts.sum() > 0:
            policy_target = visit_counts / visit_counts.sum()
        else:
            policy_target = np.zeros_like(visit_counts)
            policy_target[best_action] = 1.0

        return best_action, policy_target

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        loss_total = 0
        self.optimizer.zero_grad()
        batch = random.sample(replay_buffer, batch_size)
        for trajectory in batch:
            traj_length = len(trajectory['actions'])
            start_index = random.randint(0, traj_length)
            initial_obs = trajectory['observations'][start_index]
            initial_obs_tensor = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
            latent, value, policy_logits = self.net.initial_inference(initial_obs_tensor)

            losses = 0
            target_value = self.compute_target_value(trajectory, start_index, config.unroll_steps)
            if start_index < len(trajectory['policies']):
                target_policy = trajectory['policies'][start_index]
            else:
                target_policy = np.ones(self.action_size) / self.action_size

            target_value_tensor = torch.FloatTensor([target_value]).to(device)
            target_policy_tensor = torch.FloatTensor([target_policy]).to(device)

            value_loss = F.mse_loss(value.squeeze(), target_value_tensor)
            policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=1), target_policy_tensor, reduction='batchmean')
            step_loss = (config.value_loss_weight * value_loss +
                         config.policy_loss_weight * policy_loss)
            losses += step_loss

            current_latent = latent
            for k in range(1, config.unroll_steps + 1):
                if start_index + k - 1 < len(trajectory['actions']):
                    action = trajectory['actions'][start_index + k - 1]
                    action_tensor = torch.LongTensor([action]).to(device)
                else:
                    action_tensor = torch.LongTensor([self.action_size - 1]).to(device)

                current_latent, reward, value, policy_logits = self.net.recurrent_inference(current_latent, action_tensor)

                if start_index + k - 1 < len(trajectory['rewards']):
                    target_reward = trajectory['rewards'][start_index + k - 1]
                else:
                    target_reward = 0.0
                if start_index + k < len(trajectory['policies']):
                    target_policy = trajectory['policies'][start_index + k]
                else:
                    target_policy = np.ones(self.action_size) / self.action_size
                target_value = self.compute_target_value(trajectory, start_index + k, config.unroll_steps)

                target_reward_tensor = torch.FloatTensor([target_reward]).to(device)
                target_value_tensor = torch.FloatTensor([target_value]).to(device)
                target_policy_tensor = torch.FloatTensor([target_policy]).to(device)

                reward_loss = F.mse_loss(reward.squeeze(), target_reward_tensor)
                value_loss = F.mse_loss(value.squeeze(), target_value_tensor)
                policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=1), target_policy_tensor, reduction='batchmean')
                step_loss = (config.reward_loss_weight * reward_loss +
                             config.value_loss_weight * value_loss +
                             config.policy_loss_weight * policy_loss)
                losses += step_loss
            losses.backward()
            loss_total += losses.item()
        self.optimizer.step()
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

# --- Evaluation System with ELO Rating ---
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
                    action, _ = self.agent.select_action(obs)
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

# --- Worker Function for Parallel Self-play ---
def simulate_episode(args):
    # This worker does not initialize wandb, avoiding duplicate logins.
    agent_state_dict, board_size, latent_dim, action_size, num_simulations = args
    local_agent = MuZeroAgent(board_size, latent_dim, action_size, num_simulations)
    local_agent.net.load_state_dict(agent_state_dict)
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    total_reward = 0
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'policies': []
    }
    trajectory['observations'].append(obs)
    while not done:
        action, policy = local_agent.select_action(obs)
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
        total_reward += reward
    return trajectory, total_reward

# --- Main Training Loop with Multiprocessing ---
def main(resume, resume_file):
    board_size = config.board_size
    env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    action_size = env.action_space.n
    agent = MuZeroAgent(board_size, config.latent_dim, action_size, num_simulations=config.mcts_simulations)
    episode_count = 0
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    evaluation_interval = config.evaluation_interval
    evaluator = Evaluator(agent, env, num_eval_episodes=5)

    # If resuming, load the training state
    if resume and os.path.exists(resume_file):
        checkpoint = torch.load(resume_file, map_location=device)
        agent.net.load_state_dict(checkpoint['agent_state_dict'])
        episode_count = checkpoint.get('episode_count', 0)
        logger.info(f"Resumed training from episode {episode_count}")

    num_workers = 8
    pool = Pool(processes=num_workers)

    while episode_count < config.num_episodes:
        current_state = agent.net.state_dict()
        worker_args = [(current_state, config.board_size, config.latent_dim, action_size, config.mcts_simulations)
                       for _ in range(num_workers)]
        results = pool.map(simulate_episode, worker_args)
        for trajectory, total_reward in results:
            replay_buffer.append(trajectory)
            logger.info(f"Episode {episode_count} total reward: {total_reward}")
            wandb.log({"training_reward": total_reward, "episode": episode_count})
            episode_count += 1

            loss = agent.train(replay_buffer, batch_size=config.batch_size)
            if loss is not None:
                logger.info(f"Episode {episode_count} training loss: {loss:.4f}")
                wandb.log({"training_loss": loss, "episode": episode_count})

            if episode_count > 0 and episode_count % evaluation_interval == 0:
                avg_eval_reward, win_rate, current_elo = evaluator.evaluate(episode_count)
                logger.info(f"Evaluation after episode {episode_count}: average reward = {avg_eval_reward:.2f}, win rate = {win_rate:.2f}, ELO = {current_elo:.2f}")
                wandb.log({
                    "evaluation_avg_reward": avg_eval_reward,
                    "evaluation_win_rate": win_rate,
                    "elo_rating": current_elo,
                    "episode": episode_count
                })
                # Save periodic checkpoints including training state
                if episode_count % 10000 == 0:
                    checkpoint_path = f"muzero_model_episode_{episode_count}.pth"
                    torch.save(agent.net.state_dict(), checkpoint_path)
                    state_checkpoint = {
                        'episode_count': episode_count,
                        'agent_state_dict': agent.net.state_dict()
                    }
                    torch.save(state_checkpoint, "muzero_resume.pth")
                    logger.info(f"Model saved to '{checkpoint_path}' and training state saved to 'muzero_resume.pth'.")
                    wandb.save(checkpoint_path)

    torch.save(agent.net.state_dict(), "muzero_model_final.pth")
    logger.info("Final model saved to 'muzero_model_final.pth'.")
    wandb.save("muzero_model_final.pth")
    wandb.finish()
    pool.close()
    pool.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint")
    parser.add_argument("--resume_file", type=str, default="muzero_resume.pth", help="Path to the resume checkpoint file")
    args = parser.parse_args()
    wandb.init(project="muzero_go", config={
        "board_size": config.board_size,
        "latent_dim": config.latent_dim,
        "learning_rate": config.learning_rate,
        "mcts_simulations": config.mcts_simulations,
        "num_episodes": config.num_episodes,
        "batch_size": config.batch_size,
        "unroll_steps": config.unroll_steps,
        "discount": config.discount,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "dirichlet_alpha": config.dirichlet_alpha,
        "initial_elo": config.initial_elo,
        "elo_k": config.elo_k
    })
    main(args.resume, args.resume_file)
