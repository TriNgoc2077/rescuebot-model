import os
import yaml
import torch
from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule
from src.agents.dqn_agent import DQNAgent

def main():
    config_path = "configs/default.yaml"
    ckpt_path = "ckpts/dqn_best.pth"

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

    obs = env.reset()
    img = obs['image']
    boxes = obs['boxes']

    feat = vit_ext([img])[0]
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)

    action = agent.select_action(feat, boxes_tensor)

    print(f"Predicted action: {action}")

    env.render()

if __name__ == "__main__":
    main()
