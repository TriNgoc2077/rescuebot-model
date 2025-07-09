import os
import yaml
import torch
from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent
import cv2

def main():
    config_path = "configs/default.yaml"
    ckpt_path = "ckpts/dqn_final.pth"

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

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
    agent.epsilon = 0.0 
    # agent.epsilon = 1.0 # force agent random
    obs = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        img = obs['image']
        boxes = obs['boxes']

        feat = vit_ext([img])[0]
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)

        action = agent.select_action(feat, boxes_tensor)
        # action = 2 #force agent move forward

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Done={done}")
        # env.render()
        frame = env.render()
        cv2.imshow("MiniGrid", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)
        pos = env.env.unwrapped.agent_pos
        print(f"Pos: {pos}")

    print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
