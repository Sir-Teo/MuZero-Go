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
  python play.py
  ```