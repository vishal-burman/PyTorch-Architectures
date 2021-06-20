import numpy as np

def make_linear_regresssion_data(no_samples=10, mean=0.0, std=1.0):
    """
    Models y = 2x + 3 + e ~ Normal(0, 1)
    """
    e = np.random.normal(mean, std, no_samples)
    samples = np.arange(1, no_samples + 1, dtype=np.float32)
    samples = (2. * samples) + 3. + e
    return samples
