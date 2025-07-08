import os
import time
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent


def main(config_path="configs/default.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    writer = SummaryWriter(log_dir=cfg['log_dir'])

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
        device=device,
        lr=cfg['agent']['lr'],
        gamma=cfg['agent']['gamma'],
        epsilon_start=cfg['agent']['epsilon_start'],
        epsilon_end=cfg['agent']['epsilon_end'],
        epsilon_decay=cfg['agent']['epsilon_decay'],
        target_update_freq=cfg['agent']['target_update_freq'],
        buffer_size=cfg['agent']['buffer_size'],
        batch_size=cfg['agent']['batch_size']
    )

    global_step = 0
    episode_rewards = []

    for ep in range(1, cfg['train']['num_episodes'] + 1):
        obs = env.reset()
        feat = vit_ext([obs['image']])[0]
        boxes = torch.tensor(obs['boxes'], device=device)
        ep_reward = 0.0

        for t in range(cfg['train']['max_steps_per_episode']):
            # select and perform action
            action = agent.select_action(feat, boxes)
            next_obs, reward, done, info = env.step(action)
            next_feat = vit_ext([next_obs['image']])[0]
            next_boxes = torch.tensor(next_obs['boxes'], device=device)

            # store transition and learn
            agent.store_transition(feat, boxes, action, reward, next_feat, next_boxes, done)
            agent.learn()
            global_step += 1
            ep_reward += reward

            feat, boxes = next_feat, next_boxes

            writer.add_scalar('train/reward_step', reward, global_step)

            if done:
                break

        episode_rewards.append(ep_reward)
        avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)

        writer.add_scalar('train/episode_reward', ep_reward, ep)
        writer.add_scalar('train/avg_reward_100', avg_reward, ep)

        print(f"Episode {ep}/{cfg['train']['num_episodes']} - Reward: {ep_reward:.2f} - Avg100: {avg_reward:.2f}")

        if ep % cfg['train']['save_freq'] == 0:
            ckpt_path = os.path.join(cfg['ckpt_dir'], f"dqn_ep{ep}.pth")
            agent.save(ckpt_path)

    agent.save(os.path.join(cfg['ckpt_dir'], 'dqn_final.pth'))
    writer.close()


if __name__ == '__main__':
    main()
