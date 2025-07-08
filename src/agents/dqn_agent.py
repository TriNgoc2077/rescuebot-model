import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state_feat', 'state_boxes', 'action', 'reward', 'next_feat', 'next_boxes', 'done'))
# this module consumes ViT features + bounding-box input, outputs discrete actions.
class DQNetwork(nn.Module):
    # Simple MLP taking concatenated feature vector and flattened boxes,
    # outputting Q-values for each discrete action.
    def __init__(self, input_dim: int, action_dim: int, hidden_dims=(512, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # feat: (B, feat_dim), boxes: (B, max_boxes, 4)
        b = feat.shape[0]
        flat_boxes = boxes.view(b, -1).float()  # flatten boxes
        x = torch.cat([feat, flat_boxes], dim=1)
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        feat_dim: int,
        max_boxes: int,
        action_dim: int,
        device: torch.device = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 1e-5,
        target_update_freq: int = 1000,
        buffer_size: int = 100000,
        batch_size: int = 64
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feat_dim = feat_dim
        self.max_boxes = max_boxes
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.policy_net = DQNetwork(feat_dim + max_boxes * 4, action_dim).to(self.device)
        self.target_net = DQNetwork(feat_dim + max_boxes * 4, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, feat: torch.Tensor, boxes: torch.Tensor) -> int:
        # feat: (feat_dim,), boxes: (max_boxes,4)
        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        if sample < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                feat = feat.unsqueeze(0).to(self.device)
                boxes = boxes.unsqueeze(0).to(self.device)
                qvals = self.policy_net(feat, boxes)
                return qvals.argmax(dim=1).item()

    def store_transition(self, *args):
        # args: state_feat, state_boxes, action, reward, next_feat, next_boxes, done
        self.memory.append(Transition(*args))

    def learn(self):
        # Learning summary: 
        # 1. Get minibatch
        # 2. Calculate Q(s,a) with policy_net
        # 3. Calculate Q'(s', a') with target_net
        # 4. Calculate loss MSE: (Q - target)^2
        # 5. backward + optimizer
        # 6. Update target_net periodically
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))

        state_feat = torch.stack(batch.state_feat).to(self.device)
        state_boxes = torch.stack(batch.state_boxes).to(self.device)
        action = torch.tensor(batch.action, device=self.device).long().unsqueeze(1)
        reward = torch.tensor(batch.reward, device=self.device).float().unsqueeze(1)
        next_feat = torch.stack(batch.next_feat).to(self.device)
        next_boxes = torch.stack(batch.next_boxes).to(self.device)
        done = torch.tensor(batch.done, device=self.device).float().unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(state_feat, state_boxes).gather(1, action)
        # max Q' for next state
        with torch.no_grad():
            q_next = self.target_net(next_feat, next_boxes).max(1)[0].unsqueeze(1)
        # target: r + gamma * Q' * (1 - done)
        q_target = reward + (1 - done) * self.gamma * q_next

        loss = nn.MSELoss()(q_values, q_target)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
