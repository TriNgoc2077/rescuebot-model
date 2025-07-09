# RescueBot Model

**Vision Transformer (ViT) + Deep Reinforcement Learning for Disaster Rescue Robots**

---

## Overview

**RescueBot** is a research prototype that combines **Vision Transformers** and **Deep Q-Networks (DQN)** to train a rescue robot to:

- **Detect victims** in a simulated disaster area
- **Plan an optimal escape path** to exit the hazardous zone as quickly and safely as possible

The system is designed as a modular pipeline:

- **`envs/`**: Defines the custom simulation environments and generates RGB image observations for training.
- **`features/`**: Extracts high-level vision features using a ViT backbone.
- **`agents/`**: Implements the Deep Q-Network agent that learns to take optimal actions based on vision and environment state.
- **`utils/`**: Contains supporting modules such as the replay buffer.
- **`configs/default.yaml`**: Stores all key hyperparameters for easy tuning.
- **`logs/`**: Records training metrics for monitoring and analysis.
- **`ckpts/`**: Stores pre-trained checkpoints for quick resumption or further fine-tuning.

---

## Key Notes

- **AMD GPU support:** Uses `torch-directml` for training on AMD GPUs.
- **Limited compute:** Current training runs only cover \~2,000 episodes (out of a target 10,000). As a result, the agent may not yet generalize perfectly to all scenarios.
- **Work in progress:** This is an experimental model — we welcome contributions and improvements to enhance path planning and victim detection performance.

---

## Quick Setup

```bash
# 1. Create virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate  # On Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch-directml  # For AMD GPUs
```

---

## Training

Train the model with:

```bash
python -m src.train
```

Check and adjust hyperparameters in `configs/default.yaml` to fit your setup.

---

## Evaluate

Evaluate the agent’s performance after training:

```bash
python post_training_analysis.py
# Or
python -m src.evaluate
```

---

## Demo

- **Run a full inference episode:**

  ```bash
  python run_episode.py
  ```

- **Visualize how the agent makes decisions from random images:**

  ```bash
  python demo.py
  ```

- **Capture environment snapshots:**

  ```bash
  python snapshot_env.py
  # Images are saved to the snapshots/ directory
  ```

---

## Contributing

This is an early-stage project and we know there’s room for improvement.
Your ideas, pull requests, and feedback are highly appreciated — let’s make RescueBot smarter, faster, and more reliable **together**!

---

## Checkpoints

We provide pre-trained weights in the `ckpts/` folder as a starting point for further research or experimentation.

---

**Thank you for supporting this project!**
Let’s build safer rescue robots for the future. 💙🤖
