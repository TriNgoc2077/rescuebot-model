import os
import yaml
import torch
from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent
import cv2

def load_checkpoint(agent, path, device):
    """Load checkpoint supporting both old ('policy') and new ('model') formats."""
    ckpt = torch.load(path, map_location=device)
    state_dict = (ckpt.get('policy') or ckpt.get('model') or ckpt)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    print(f"Loaded weights from {path}")

def main():
    config_path = "configs/default.yaml"
    best = "ckpts/dqn_best.pth"
    ckpt_path = best if os.path.exists(best) else "ckpts/dqn_final.pth"
    # ckpt_path = "ckpts/dqn_final.pth"

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    load_checkpoint(agent, ckpt_path, device)
    agent.epsilon = 0.0

    obs = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        img = obs['image']
        boxes = obs['boxes']

        feat = vit_ext([img])[0]                  
        boxes_tensor = torch.tensor(boxes, device=device)

        action = agent.select_action(feat, boxes_tensor)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Done={done}")
        frame = env.render()
        cv2.imshow("MiniGrid", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        pos = env.env.unwrapped.agent_pos
        print(f"Pos: {pos}")

    print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
