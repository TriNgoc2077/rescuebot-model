import os
import time
import yaml
import torch
import glob
import re
from torch.utils.tensorboard import SummaryWriter

from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint file in the checkpoint directory"""
    if not os.path.exists(ckpt_dir):
        return None
    
    # Look for checkpoint files with pattern dqn_ep*.pth
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "dqn_ep*.pth"))
    if not ckpt_files:
        # Also check for final checkpoint
        final_ckpt = os.path.join(ckpt_dir, "dqn_final.pth")
        if os.path.exists(final_ckpt):
            return final_ckpt, None
        return None, None
    
    # Extract episode numbers and find the latest
    episode_numbers = []
    for ckpt_file in ckpt_files:
        match = re.search(r'dqn_ep(\d+)\.pth', ckpt_file)
        if match:
            episode_numbers.append((int(match.group(1)), ckpt_file))
    
    if episode_numbers:
        episode_numbers.sort(key=lambda x: x[0])
        latest_ep, latest_file = episode_numbers[-1]
        return latest_file, latest_ep
    
    return None, None


def load_training_state(ckpt_path):
    """Load training state from checkpoint"""
    if not os.path.exists(ckpt_path):
        return None
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return None


def save_training_state(agent, episode, global_step, episode_rewards, ckpt_path):
    """Save complete training state"""
    checkpoint = {
        'episode': episode,
        'global_step': global_step,
        'episode_rewards': episode_rewards,
    }
    
    # Save agent using its own save method and merge with training state
    agent.save(ckpt_path)
    
    # Load the saved agent checkpoint and add training metadata
    agent_checkpoint = torch.load(ckpt_path, map_location='cpu')
    agent_checkpoint.update(checkpoint)
    
    # Save the combined checkpoint
    torch.save(agent_checkpoint, ckpt_path)


def main(config_path="configs/default.yaml", resume=True):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    # Initialize environment and feature extractor
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

    # Initialize agent
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

    # Initialize training variables
    start_episode = 1
    global_step = 0
    episode_rewards = []

    # Try to load checkpoint if resume is True
    if resume:
        latest_ckpt, latest_ep = find_latest_checkpoint(cfg['ckpt_dir'])
        if latest_ckpt:
            print(f"Found checkpoint: {latest_ckpt}")
            checkpoint = load_training_state(latest_ckpt)
            
            if checkpoint:
                # Load agent using its own load method
                agent.load(latest_ckpt)
                
                # Load training state - handle both old and new checkpoint formats
                if 'episode' in checkpoint:
                    # New format checkpoint with training metadata
                    start_episode = checkpoint['episode'] + 1
                    global_step = checkpoint.get('global_step', 0)
                    episode_rewards = checkpoint.get('episode_rewards', [])
                    print(f"Resumed training from episode {start_episode-1} (new format)")
                elif latest_ep is not None:
                    # Old format checkpoint - infer from filename
                    start_episode = latest_ep + 1
                    global_step = latest_ep * cfg['train']['max_steps_per_episode']  # Estimate global step
                    episode_rewards = [0.0] * latest_ep  # Dummy rewards history
                    print(f"Resumed training from episode {latest_ep} (old format - estimated)")
                else:
                    # Fallback
                    start_episode = 1
                    global_step = 0
                    episode_rewards = []
                    print("Could not determine episode number, starting from episode 1")
                
                print(f"Starting episode: {start_episode}")
                print(f"Global step: {global_step}")
                print(f"Episode rewards history: {len(episode_rewards)} episodes")
            else:
                print("Failed to load checkpoint, starting from scratch")
        else:
            print("No checkpoint found, starting from scratch")
    else:
        print("Resume disabled, starting from scratch")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # Training loop
    for ep in range(start_episode, cfg['train']['num_episodes'] + 1):
        obs = env.reset()
        feat = vit_ext([obs['image']])[0]
        boxes = torch.tensor(obs['boxes'], device=device)
        ep_reward = 0.0

        for t in range(cfg['train']['max_steps_per_episode']):
            # Select and perform action
            action = agent.select_action(feat, boxes)
            next_obs, reward, done, info = env.step(action)
            next_feat = vit_ext([next_obs['image']])[0]
            next_boxes = torch.tensor(next_obs['boxes'], device=device)

            # Store transition and learn
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
        writer.add_scalar('train/epsilon', agent.epsilon if hasattr(agent, 'epsilon') else 0, ep)

        print(f"Episode {ep}/{cfg['train']['num_episodes']} - Reward: {ep_reward:.2f} - Avg100: {avg_reward:.2f}")

        # Save checkpoint
        if ep % cfg['train']['save_freq'] == 0:
            ckpt_path = os.path.join(cfg['ckpt_dir'], f"dqn_ep{ep}.pth")
            save_training_state(agent, ep, global_step, episode_rewards, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = os.path.join(cfg['ckpt_dir'], 'dqn_final.pth')
    save_training_state(agent, cfg['train']['num_episodes'], global_step, episode_rewards, final_ckpt_path)
    print(f"Final checkpoint saved: {final_ckpt_path}")
    
    writer.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN Agent with Checkpoint Support')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='Path to config file')
    parser.add_argument('--no-resume', action='store_true', 
                       help='Start training from scratch instead of resuming')
    
    args = parser.parse_args()
    
    main(config_path=args.config, resume=not args.no_resume)