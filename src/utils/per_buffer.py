import numpy as np
import random

class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    Used here to sample transitions proportionally to their priority.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Stores transitions and samples them according to their TD-error priority.
    """
    def __init__(self, capacity, batch_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.batch_size = batch_size
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, transition):
        # Use max priority for new transitions
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, transition)

    append = push  # for compatibility

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []
        # Increase beta linearly
        beta = min(1.0, self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors, eps=1e-6):
        for idx, error in zip(idxs, errors):
            p = (abs(error) + eps) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries