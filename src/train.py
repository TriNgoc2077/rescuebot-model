import os
import yaml
import glob
import re
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent

# Use Prioritized Experience Replay if available
try:
    from src.utils.per_buffer import PrioritizedReplayBuffer as ReplayBuffer
    USE_PER = True
except ImportError:
    from src.utils.replay_buffer import ReplayBuffer
    USE_PER = False


def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    if not files:
        return None
    def extract_ep(path):
        m = re.search(r'dqn_ep(\d+)\.pth', os.path.basename(path))
        return int(m.group(1)) if m else -1
    files = sorted(files, key=lambda p: (extract_ep(p), os.path.getmtime(p)), reverse=True)
    return files[0]


def save_training_state(agent, episode, global_step, rewards, ckpt_path, optimizer=None, scheduler=None):
    torch.save(agent.policy_net.state_dict(), ckpt_path)
    meta = {'episode': episode, 'global_step': global_step, 'episode_rewards': rewards}
    if optimizer is not None:
        meta['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        meta['scheduler_state'] = scheduler.state_dict()
    torch.save({'model': torch.load(ckpt_path, map_location='cpu'), **meta}, ckpt_path)


def main(cfg_path="configs/default.yaml", resume=True):
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    # Environment and feature extractor
    env = RescueEnv(cfg['env']['name'], cfg['env']['img_size'], cfg['env']['max_boxes'])
    vit_ext = ViTFeatureExtractorModule(
        model_name=cfg['vit']['model_name'],
        pretrained=cfg['vit']['pretrained'],
        freeze_backbone=cfg['vit']['freeze_backbone'],
        device=device
    )

    # Agent
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
        batch_size=cfg['agent']['batch_size'],
        n_step=cfg['agent'].get('n_step', 3)
    )

    # Replay buffer
    if USE_PER:
        print("Using Prioritized Experience Replay")
        agent.memory = ReplayBuffer(
            capacity=cfg['agent']['buffer_size'],
            batch_size=cfg['agent']['batch_size'],
            alpha=cfg['agent'].get('per_alpha', 0.6),
            beta_start=cfg['agent'].get('per_beta_start', 0.4),
            beta_frames=cfg['train']['num_episodes'] * cfg['train']['max_steps_per_episode']
        )
    else:
        agent.memory = ReplayBuffer(
            capacity=cfg['agent']['buffer_size'],
            batch_size=cfg['agent']['batch_size']
        )

    # Scheduler: Cosine annealing
    scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=cfg['train']['num_episodes'],
        eta_min=cfg['agent'].get('lr_min', 1e-6)
    )

    # Resume checkpoint
    start_ep, global_step, rewards = 1, 0, []
    if resume:
        ckpt = find_latest_checkpoint(cfg['ckpt_dir'])
        if ckpt:
            data = torch.load(ckpt, map_location=device)
            agent.load(ckpt)
            if 'optimizer_state' in data:
                try:
                    agent.optimizer.load_state_dict(data['optimizer_state'])
                    print("✅ Loaded optimizer state")
                except Exception as e:
                    print(f"⚠️ Skipping optimizer load: {e}")
            if 'scheduler_state' in data:
                try:
                    scheduler.load_state_dict(data['scheduler_state'])
                except Exception:
                    pass
            start_ep = data.get('episode', 0) + 1
            global_step = data.get('global_step', 0)
            rewards = data.get('episode_rewards', [])
            print(f"Resumed from {ckpt}: ep {start_ep-1}, step {global_step}")
        else:
            print("No checkpoint found, starting fresh.")

    writer = SummaryWriter(cfg['log_dir'])
    best_avg = max(rewards[-100:]) if rewards else -float('inf')

    # Training loop
    for ep in range(start_ep, cfg['train']['num_episodes'] + 1):
        obs = env.reset()
        feat = vit_ext([obs['image']])[0]
        boxes = torch.tensor(obs['boxes'], device=device)
        ep_reward = 0.0

        for t in range(cfg['train']['max_steps_per_episode']):
            action = agent.select_action(feat, boxes)
            next_obs, reward, done, _ = env.step(action)
            # reward clipping
            reward = max(min(reward, 1.0), -1.0)
            next_feat = vit_ext([next_obs['image']])[0]
            next_boxes = torch.tensor(next_obs['boxes'], device=device)

            agent.store_transition(feat, boxes, action, reward, next_feat, next_boxes, done)
            ret = agent.learn()
            if ret:
                loss, q_max, q_min = ret
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/q_max', q_max, global_step)
                writer.add_scalar('train/q_min', q_min, global_step)

            feat, boxes = next_feat, next_boxes
            ep_reward += reward
            global_step += 1

            if done:
                break

        # Scheduler & logging
        scheduler.step()
        lr = agent.optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', lr, ep)
        writer.add_scalar('train/episode_reward', ep_reward, ep)
        avg100 = (sum(rewards[-99:]) + ep_reward) / min(len(rewards) + 1, 100)
        writer.add_scalar('train/avg_reward_100', avg100, ep)
        writer.add_scalar('train/epsilon', agent.epsilon, ep)
        print(f"Ep {ep}/{cfg['train']['num_episodes']} | R: {ep_reward:.2f} | Av100: {avg100:.2f} | LR: {lr:.5f}")

        # Save best
        if avg100 > best_avg:
            best_avg = avg100
            agent.save(os.path.join(cfg['ckpt_dir'], 'dqn_best.pth'))
            print(f"New best at ep {ep} (avg100 {avg100:.2f})")

        # Periodic save
        if ep % cfg['train']['save_freq'] == 0:
            path = os.path.join(cfg['ckpt_dir'], f"dqn_ep{ep}.pth")
            save_training_state(agent, ep, global_step, rewards, path,
                                 optimizer=agent.optimizer, scheduler=scheduler)
            print(f"Saved checkpoint {path}")
        rewards.append(ep_reward)

    # Final save
    final_path = os.path.join(cfg['ckpt_dir'], 'dqn_final.pth')
    save_training_state(agent, cfg['train']['num_episodes'], global_step, rewards,
                         final_path, optimizer=agent.optimizer, scheduler=scheduler)
    print("Training complete.")
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/default.yaml')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    main(cfg_path=args.config, resume=not args.no_resume)