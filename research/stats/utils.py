import numpy as np

def make_linear_regresssion_data(no_samples=10, mean=0.0, std=1.0):
    """
    Models y = 2x + 3 + e ~ Normal(0, 1)
    """
    e = np.random.normal(mean, std, no_samples)
    inputs = np.arange(1, no_samples + 1, dtype=np.float32)
    labels = (2. * inputs) + 3. + e
    return inputs, labels, e

def rss(errors):
    """
    Returns Residual Sum of Squares for error-list
    """
    rss_result = np.sum(np.square(errors))
    return rss_result

def minimize_rss(inputs, labels):
    """
    Returns predicted B0 and B1, calculated through minimization of RSS
    """
    B1 = np.sum((inputs - np.mean(inputs)) * (labels - np.mean(labels))) / np.sum(np.square(inputs - np.mean(inputs)))
    
    B0 = np.mean(labels) - (B1 * np.mean(inputs))
    return B1, B0
