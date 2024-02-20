import numpy as np

class CustomDistribution:
    def __init__(self, seed=None):
        self.random_state = np.random.default_rng(seed)

    def next_exponential(self, scale):
        return self.random_state.exponential(scale)

    def next_double_in_interval(self, min_val, max_val):
        return self.random_state.uniform(min_val, max_val)

