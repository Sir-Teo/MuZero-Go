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
    latent_dim = 64
    learning_rate = 1e-5
    mcts_simulations = 50
    num_episodes = 100000
    num_envs = 32       # Number of parallel environments
    replay_buffer_size = 10000
    batch_size = 64
    unroll_steps = 64
    discount = 0.99
    value_loss_weight = 1.0
    policy_loss_weight = 1.0
    reward_loss_weight = 1.0
    dirichlet_epsilon = 0.02
    dirichlet_alpha = 0.01
    initial_elo = 1000
    elo_k = 32
    evaluation_interval = 1

config = Config()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def evaluate(self):
        current_wins = 0
        # Alternate which agent starts to balance first-move advantages
        for i in range(self.num_games):
            starting_player = i % 2  # Alternate starting players
            result = self.play_game(starting_player=starting_player)
            # If best_agent starts (turn=1), then a win (reward > 0) means best_agent won,
            # so invert the outcome for current_agent’s perspective.
            if starting_player == 1:
                result = 1 - result
            current_wins += result

        win_rate = current_wins / self.num_games

        # Update Elo ratings for both agents using the Elo update formula:
        expected_score = 1 / (1 + 10 ** ((self.best_elo - self.current_elo) / 400))
        self.current_elo += config.elo_k * (win_rate - expected_score)
        self.best_elo += config.elo_k * ((1 - win_rate) - (1 - expected_score))

        return win_rate, self.current_elo



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

            # Inside BatchedMCTS.run(), in the simulation loop:
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
                    is_terminal = (reward != 0)  # Adjust terminal condition if needed
                    child_node = MCTSNode(next_latent, prior=0, terminal=is_terminal)
                    
                    # Apply valid move mask to child policy (unchanged)
                    next_obs = observations[env_idx]
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
                    # Incorporate reward into the backup value
                    backup_value = reward + config.discount * value
                    leaf.children[action]['node'] = child_node
                    self.backpropagate(path + [child_node], backup_value)
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
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)

        losses = 0
        with torch.cuda.amp.autocast():  # Updated API
            for trajectory in batch:
                traj_length = len(trajectory['actions'])
                start_index = random.randint(0, traj_length - 1) 
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

class SelfPlayEvaluator:
    def __init__(self, current_agent, best_agent, env, num_games=20):
        self.current_agent = current_agent
        self.best_agent = best_agent  # This is a snapshot of the best model so far
        self.env = env
        self.num_games = num_games
        # Initialize Elo ratings; both start with the same rating
        self.current_elo = config.initial_elo
        self.best_elo = config.initial_elo

    def play_game(self, starting_player=0):
        """
        Play a single game between current_agent and best_agent.
        Alternate moves based on starting_player (0: current starts, 1: best starts).
        Each agent uses its own MCTS search.
        """
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        turn = starting_player
        while not done:
            if turn == 0:
                # current_agent's move
                mcts = BatchedMCTS(self.current_agent.net, 1, self.current_agent.action_size, self.current_agent.mcts_simulations)
            else:
                # best_agent's move
                mcts = BatchedMCTS(self.best_agent.net, 1, self.best_agent.action_size, self.best_agent.mcts_simulations)
            roots = mcts.run(np.expand_dims(obs, 0))
            root = roots[0]
            # Get valid moves mask
            invalid_moves = obs[govars.INVD_CHNL]
            valid_mask = (invalid_moves.flatten() == 0).astype(np.float32)
            valid_mask = np.concatenate([valid_mask, np.array([1.0])])
            visit_counts = np.array([child['visit_count'] if valid_mask[a] > 0 else 0 
                                     for a, child in root.children.items()])
            if visit_counts.sum() > 0:
                action = int(np.argmax(visit_counts))
            else:
                valid_actions = np.nonzero(valid_mask)[0]
                action = int(random.choice(valid_actions))
            # Execute action in environment
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
                done = done or truncated
            else:
                obs, reward, done, info = result
            if isinstance(obs, tuple):
                obs = obs[0]
            # Alternate turn after each move
            turn = 1 - turn
        return 1 if info.get('winner') == (0 if starting_player == 0 else 1) else 0

    def evaluate(self):
        current_wins = 0
        # Alternate which agent starts to balance first-move advantages
        for i in range(self.num_games):
            starting_player = i % 2  # Alternate starting players
            result = self.play_game(starting_player=starting_player)
            # If best_agent starts (turn=1), then a win (reward > 0) means best_agent won,
            # so invert the outcome for current_agent’s perspective.
            if starting_player == 1:
                result = 1 - result
            current_wins += result

        win_rate = current_wins / self.num_games

        # Update Elo ratings for both agents using the Elo update formula:
        expected_score = 1 / (1 + 10 ** ((self.best_elo - self.current_elo) / 400))
        self.current_elo += config.elo_k * (win_rate - expected_score)
        self.best_elo += config.elo_k * ((1 - win_rate) - (1 - expected_score))

        return win_rate, self.current_elo


# Main Training Loop
def main():
    board_size = config.board_size
    env_action_size = board_size * board_size + 1
    agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    # Create a separate best agent (deep copy) for evaluation:
    best_agent = MuZeroAgent(board_size, config.latent_dim, env_action_size, config.mcts_simulations)
    best_agent.net.load_state_dict(agent.net.state_dict())
    
    vec_env = VectorGoEnv(config.num_envs, board_size)
    batched_mcts = BatchedMCTS(agent.net, config.num_envs, env_action_size, config.mcts_simulations)
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    # Use the same Go environment for evaluation matches
    eval_env = gym.make("gym_go:go-v0", size=board_size, komi=0, reward_method='real')
    selfplay_evaluator = SelfPlayEvaluator(agent, best_agent, eval_env, num_games=20)
    
    episode_count = 0
    start_time = time.time()  # To track elapsed time

    while episode_count < config.num_episodes:
        observations = vec_env.reset()
        done = [False] * config.num_envs
        trajectories = [{'observations': [obs], 'actions': [], 'rewards': [], 'policies': []} 
                        for obs in observations]
        new_episodes = 0

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
                        policy = np.zeros(env_action_size)
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
                        logger.info(f"Episode {episode_count} completed | Elapsed Time: {elapsed:.2f}s")
                        wandb.log({"episode_complete": episode_count, "elapsed_time": elapsed})

            observations = next_observations

        # Training updates for this batch of new episodes
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
            logger.info(f"Batch Training Summary | Episodes: {new_episodes} | Avg Loss: {avg_loss:.4f}")

        # Run self-play evaluation at fixed intervals
        if episode_count > 0 and episode_count % config.evaluation_interval == 0:
            win_rate, new_elo = selfplay_evaluator.evaluate()
            logger.info(f"Self-play Evaluation at Episode {episode_count} | Win Rate: {win_rate:.2f} | Elo: {new_elo:.2f}")
            wandb.log({"selfplay_win_rate": win_rate, "selfplay_elo": new_elo, "episode": episode_count})
            # If the current agent outperforms the best agent by a threshold, update best_agent.
            if win_rate > 0.55:  # Adjust the threshold as needed.
                best_agent.net.load_state_dict(agent.net.state_dict())
                logger.info("Best agent updated with current agent's parameters.")

            # Optionally, save models at intervals.
            if episode_count % 10000 == 0:
                model_path = f"muzero_model_episode_{episode_count}.pth"
                torch.save(agent.net.state_dict(), model_path)
                wandb.save(model_path)

    # Final save
    torch.save(agent.net.state_dict(), "muzero_model_final.pth")
    wandb.save("muzero_model_final.pth")
    wandb.finish()


if __name__ == "__main__":
    wandb.init(project="muzero_go_vectorized", config=vars(config))
    logger.info("Starting training process...")
    main()
    logger.info("Training completed.")