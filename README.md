# MuZero-Go

A MuZero-based Go training framework leveraging the GymGo environment.

## Installation

1. Install GymGo (Go environment for OpenAI Gym):
   ```bash
   git clone https://github.com/huangeddie/GymGo.git
   cd GymGo
   pip install -e .
   cd ..
   ```

2. Install MuZero-Go dependencies:
   ```bash
   pip install torch numpy gym wandb
   ```

## Usage

- Train the model:
  ```bash
  python main.py
  ```

- Play against the trained agent:
  ```bash
  # By default, loads 'muzero_model_final.pth' or a specific checkpoint if provided
  python play.py [path/to/model_checkpoint.pth]
  ```

- Generate self-play game data:
  ```bash
  # Uses a specified checkpoint to generate game data
  python self_play.py --weights [path/to/model_checkpoint.pth] --num_games 100 --output_dir self_play_data
  ```

## Checkpoints

- During training (`main.py`), model checkpoints are saved periodically in the `checkpoints/<run_id>/` directory.
- The final model is saved as `muzero_model_final.pth` within the run's checkpoint directory.
- You can specify a checkpoint file for `play.py` and `self_play.py` as shown in the Usage section.

## Model Architecture

The core of the agent is based on the MuZero algorithm, which consists of three main neural network components:

1.  **Representation Network:** Takes the current board state (represented as multiple input planes) and encodes it into a lower-dimensional hidden state (latent representation).
2.  **Dynamics Network:** Given a hidden state and an action, this network predicts the *next* hidden state and the immediate reward received for taking that action. This allows the model to simulate future steps internally without needing the real environment.
3.  **Prediction Network:** Takes a hidden state and predicts two things:
    *   The policy (a probability distribution over possible next moves).
    *   The value (an estimate of the expected future outcome from that state).

These networks work together within a Monte Carlo Tree Search (MCTS) to plan and select the best moves.

## Training Process

The model is trained using the following steps:

1.  **Self-Play:** The current best version of the agent plays games against itself. During these games:
    *   MCTS is used at each step to select a move. The search explores potential future game states using the dynamics and prediction networks.
    *   The visit counts from the MCTS search provide the training target for the policy network.
    *   Game data (observations, actions, rewards, policies) is stored in a replay buffer.
2.  **Training:**
    *   Batches of game data are sampled from the replay buffer.
    *   The networks are trained to better predict the policy targets, observed rewards, and eventual game outcomes (value targets) recorded during self-play.
    *   The representation network learns to create useful hidden states, the dynamics network learns to predict transitions and rewards, and the prediction network learns to predict policies and values from those states.
3.  **Evaluation:** Periodically, the currently training agent is evaluated against the previous best agent. If it wins significantly more often, its weights become the new "best" weights used for future self-play.

This cycle of self-play, training, and evaluation allows the agent to gradually improve its understanding of Go strategy.