import numpy as np
from .module import Module
from wings.nn.parameter import Parameter

class Linear(Module):
    """
    Implement a linear fully connected layer.
    """
    
    def __init__(self, in_features, out_features, 
            init_strategy='he', scale=None) -> None:
        """
        Initialize a fully connected linear layer.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            init_strategy (str): 'he' -> He initialization
                                 'xe' -> Xavier initialization
                                 'norm' -> random initialization
                                 'scale' -> variance with random initialization
        """


        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(self.in_features, self.out_features))
        # bias initialized to zeros by default
        self.bias = Parameter(np.zeros(self.out_features))

        # reset the parameters
        self.reset_parameters(init_strategy, scale)
        # append parameters to _parameters
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias

    def reset_parameters(self, init_strategy, scale):
        """
        Resets the model parameters according to requested initialization
        strategy.
        """
        if init_strategy == 'he':
            self.weight.data *= np.sqrt(2 / self.in_features)
        elif init_strategy == 'xe':
            self.weight.data *= np.sqrt(1 / self.in_features)
        elif init_strategy == 'norm':
            self.weight.data *= scale
        else:
            raise ValueError("Invalid init strategy")

    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        return f"Linear(in_feature={self.in_features}, out_features=" \
            f"{self.out_features})"

    def forward(self, X):
        self.X = X
        return X @ self.weight.data + self.bias.data

    def backward(self, dout):
        self.bias.grad = np.sum(dout, axis=0)
        self.weight.grad = self.X.T @ dout
        return dout @ self.weight.data.T



