import torch
from src.agents.dqn_agent import DQNAgent
from src.envs.rescue_env import RescueEnv
from src.features.vit_extractor import ViTFeatureExtractorModule

env = RescueEnv()
vit_ext = ViTFeatureExtractorModule(freeze_backbone=True)

# init agent  
agent = DQNAgent(
    feat_dim=768,
    max_boxes=5,
    action_dim=env.action_space.n,
    lr=1e-4,
    gamma=0.99
)
print(agent.policy_net)

# make a decision
obs = env.reset()
feat = vit_ext([obs["image"]])[0]        
boxes = torch.tensor(obs["boxes"])        # (max_boxes,4)
action = agent.select_action(feat, boxes)
print("Action:", action)

# learning 
for _ in range(100):
    boxes = torch.tensor(obs["boxes"])
    agent.store_transition(
        feat, boxes, action, 0.0, feat, boxes, False
    )
agent.learn()