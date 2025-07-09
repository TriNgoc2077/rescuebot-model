import random
from collections import deque

class ReplayBuffer(deque):
    """
    Uniform replay buffer; PER handled in train.py.
    """
    def __init__(self, capacity: int, batch_size: int):
        super().__init__(maxlen=capacity)
        self.batch_size = batch_size

    def append(self, transition):
        super().append(transition)

    def sample(self):
        return random.sample(self, self.batch_size)

    def __len__(self):
        return super().__len__()