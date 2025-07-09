import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.utils.per_buffer import PrioritizedReplayBuffer

Transition = namedtuple('Transition', (
    'state_feat','state_boxes','action','reward','next_feat','next_boxes','done'
))

class DuelingDQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=(512,256)):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()
        )
        self.value_head = nn.Linear(hidden_dims[1], 1)
        self.adv_head  = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, feat, boxes):
        b = feat.size(0)
        flat = boxes.view(b, -1).float()
        x = torch.cat([feat, flat], dim=1)
        x = self.input_norm(x)
        h = self.shared(x)
        V = self.value_head(h)
        A = self.adv_head(h)
        return V + (A - A.mean(dim=1, keepdim=True))

class DQNAgent:
    def __init__(
        self, feat_dim, max_boxes, action_dim, device,
        lr=1e-4, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1e-5,
        target_update_freq=1000, buffer_size=100000, batch_size=64, n_step=3
    ):
        self.device = device
        input_dim = feat_dim + max_boxes * 4
        self.policy_net = DuelingDQNetwork(input_dim, action_dim).to(device)
        self.target_net = DuelingDQNetwork(input_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.memory = None  # set in train.py
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # multi-step buffer
        self.n_step = n_step
        self.n_gamma = gamma
        self.n_buffer = deque(maxlen=self.n_step)

    def select_action(self, feat, boxes):
        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        if sample < self.epsilon:
            return random.randrange(self.policy_net.adv_head.out_features)
        with torch.no_grad():
            q = self.policy_net(feat.unsqueeze(0).to(self.device),
                                 boxes.unsqueeze(0).to(self.device))
        return q.argmax(dim=1).item()

    def store_transition(self, s_f, s_b, a, r, n_f, n_b, d):
        # n-step transition accumulation
        trans = Transition(s_f.detach(), s_b.detach(), a, r, n_f.detach(), n_b.detach(), d)
        self.n_buffer.append(trans)
        if len(self.n_buffer) < self.n_step:
            return
        # compute n-step return
        Rn, done_n = 0.0, False
        for i, tr in enumerate(self.n_buffer):
            Rn += (self.n_gamma ** i) * tr.reward
            if tr.done:
                done_n = True
                break
        s0 = self.n_buffer[0]
        sn = self.n_buffer[-1]
        # store aggregated transition
        self.memory.append(
            Transition(s0.state_feat, s0.state_boxes, s0.action,
                       Rn, sn.next_feat, sn.next_boxes, done_n)
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        if isinstance(self.memory, PrioritizedReplayBuffer):
            batch, idxs, is_weights = self.memory.sample()
        else:
            batch = random.sample(self.memory, self.batch_size)
            idxs, is_weights = None, None

        batch = Transition(*zip(*batch))
        sf = torch.stack(batch.state_feat).to(self.device)
        sb = torch.stack(batch.state_boxes).to(self.device)
        a  = torch.tensor(batch.action, device=self.device).long().unsqueeze(1)
        r  = torch.tensor(batch.reward, device=self.device).float().unsqueeze(1)
        nf = torch.stack(batch.next_feat).to(self.device)
        nb = torch.stack(batch.next_boxes).to(self.device)
        d  = torch.tensor(batch.done, device=self.device).float().unsqueeze(1)

        q_pred = self.policy_net(sf, sb).gather(1, a)
        with torch.no_grad():
            next_actions = self.policy_net(nf, nb).argmax(1, keepdim=True)
            q_next = self.target_net(nf, nb).gather(1, next_actions)
        q_target = r + (1 - d) * (self.gamma**self.n_step) * q_next

        if idxs is not None:
            losses = F.smooth_l1_loss(q_pred, q_target, reduction='none')
            is_w = torch.tensor(is_weights, device=self.device).float().unsqueeze(1)
            loss = (losses * is_w).mean()
            errors = (losses.detach().abs().cpu().numpy().flatten().tolist())
            self.memory.update_priorities(idxs, errors)
        else:
            loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[TargetNet] Updated at step {self.steps_done}")

        qs = q_pred.detach()
        return loss.item(), qs.max().item(), qs.min().item()

    def save(self, path):
        torch.save({'policy': self.policy_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epsilon': self.epsilon}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        old_sd = ckpt.get('policy', ckpt.get('model', ckpt))
        new_sd = {}
        for k, v in old_sd.items():
            if k.startswith('net.'):
                new_sd[k.replace('net.', 'shared.')] = v
            else:
                new_sd[k] = v
        self.policy_net.load_state_dict(new_sd, strict=False)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = ckpt.get('epsilon', self.epsilon)
