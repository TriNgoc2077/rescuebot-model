"""
Evaluation script for trained RescueBot DQN agent.
Run episodes and report average reward (and optionally render).
"""
import os
import argparse
import yaml
import torch
import numpy as np

from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent

def evaluate(cfg, ckpt_path, num_episodes, render):
    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    env = RescueEnv(
        env_name=cfg['env']['name'],
        img_size=cfg['env']['img_size'],
        max_boxes=cfg['env']['max_boxes']
    )

    vit_ext = ViTFeatureExtractorModule(
        model_name=cfg['vit']['model_name'],
        pretrained=cfg['vit']['pretrained'],
        freeze_backbone=cfg['vit']['freeze_backbone'],
        device=device
    )

    agent = DQNAgent(
        feat_dim=vit_ext.hidden_dim,
        max_boxes=cfg['env']['max_boxes'],
        action_dim=env.action_space.n,
        device=device
    )

    agent.load(ckpt_path)
    agent.policy_net.eval()

    returns = []
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        feat = vit_ext([obs['image']])[0]
        boxes = torch.tensor(obs['boxes'], device=device)
        ep_reward = 0.0
        done = False

        eps_backup = agent.epsilon
        agent.epsilon = 0.0

        while not done:
            action = agent.select_action(feat, boxes)
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()

            feat = vit_ext([next_obs['image']])[0]
            boxes = torch.tensor(next_obs['boxes'], device=device)

        agent.epsilon = eps_backup

        returns.append(ep_reward)
        print(f"Episode {ep}/{num_episodes} - Return: {ep_reward:.2f}")

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nEvaluation over {num_episodes} episodes: Avg Return = {avg_return:.2f} \u00B1 {std_return:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RescueBot DQN agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to agent checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.ckpt is None:
        args.ckpt = os.path.join(cfg['ckpt_dir'], 'dqn_final.pth')

    evaluate(cfg, args.ckpt, args.episodes, args.render)

if __name__ == '__main__':
    main()
