import random
from collections import deque


class ReplayBuffer(deque):
    """
    Replay buffer implemented as a deque with fixed capacity.
    Supports random sampling of fixed batch size.
    """
    def __init__(self, capacity: int, batch_size: int):
        """Initialize the buffer with maximum capacity and batch size."""
        super().__init__(maxlen=capacity)
        self.batch_size = batch_size

    def append(self, transition):
        """Add a new transition to the buffer."""
        super().append(transition)

    def sample(self):
        """
        Sample a batch of transitions.

        Returns:
            List of transitions of length self.batch_size.
        """
        return random.sample(self, self.batch_size)

    def __len__(self):
        """Return current size of buffer."""
        return super().__len__()
