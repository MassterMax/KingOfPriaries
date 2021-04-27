import collections
import numpy as np


class ExperienceReplay:
    def __init__(self, capacity, n_frames):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_frames = n_frames

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, is_done = zip(*[self.buffer[idx] for idx in indices])

        return states, actions, rewards, next_states, is_done
