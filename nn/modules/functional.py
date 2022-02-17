import cupy as np

def sigmoid(X):
    """
    Implements a numerically stable sigmoid function.

    Args:
        X (array): input

    Returns:
        array of same shape as X
    """
    # clip to prevent underflow
    X = np.clip(X, -700, 700)
    mask = (X >= 0)
    res = np.empty_like(X)
    # calculate sigmoid for X >= 0
    res[mask] =  1 / (1 + np.exp(-X[mask]))
    # calculate sigmoid for X < 0
    t = np.exp(X[~mask])
    res[~mask] = t / (1 + t)
    return res

def sigmoid_derivative(X):
    """
    Calculates derivative of sigmoid(X)

    Args:
        X (array): input

    Returns:
        array of same shape as X
    """
    sig_x = sigmoid(X)
    return sig_x * (1 - sig_x)

