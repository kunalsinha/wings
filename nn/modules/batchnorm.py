from .module import Module
from wings.nn.parameter import Parameter
import numpy as np


class BatchNorm(Module):
    """
    Implements BatchNorm layer.
    """

    def __init__(self, num_features, momentum=0.9):
        """
        Args:
            num_features: D from an input of size (N, D)
            momentum: coefficient used to calculate the running avg of mean
                and std deviation.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-8
        self.momentum = momentum
        # initialize gamma to ones
        self.gamma = Parameter(np.ones(num_features))
        # initialize beta to zeros
        self.beta = Parameter(np.zeros(num_features))
        # append parameters to _parameters
        self._parameters['gamma'] = self.gamma
        self._parameters['beta'] = self.beta
        # initialize running_mean and running_var
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)

    def forward(self, X):
        """
        Implements forward pass of the BatchNorm layer.
        """
        self.N, self.D = X.shape
        if self.D != self.num_features:
            raise ValueError(
                "Mismatch in number of features between X and batchnorm layer")
        self.x = X
        if self._mode == "train":
            # calculate the mean of every feature
            self.mu = np.mean(self.x, axis=0)
            # calculate the variance of every feature
            self.var = np.var(self.x, axis=0)
            # update the running averages
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * self.var
        else:
            self.mu = self.running_mean
            self.var = self.running_var
        # calculate the std deviation of every feature
        # add eps to prevent divide by zero while calculating (x - mu)/std
        self.std = np.sqrt(self.var + self.eps)
        # calculate x-mu
        self.xmmu = self.x - self.mu
        # calculate x_norm
        self.x_norm = self.xmmu / self.std
        # shift and scale x_norm by gamma and beta
        out = self.gamma.data * self.x_norm + self.beta.data
        return out

    def backward(self, dout):
        """
        Implements backward pass of the BatchNorm layer. Follows the
        computational graph backward to compute the derivatives with respect
        to gamma, beta and the input.

        Args:
            dout: gradient from the upstream.
        """
        # compute the gradient wrt beta
        self.beta.grad = np.sum(dout, axis=0)
        dgxnorm = dout
        dxnorm = dgxnorm * self.gamma.data
        # compute the gradient wrt gamma
        self.gamma.grad = np.sum(dgxnorm * self.x_norm, axis=0)
        dinvstd = np.sum(dxnorm * self.xmmu, axis=0)
        dstd = dinvstd * (-1 / (self.std ** 2))
        dvar = dstd * (0.5 / self.std)
        dxmmu2 = dvar * np.ones((self.N, self.D)) * (1 / self.N)
        dxmmu = 2 * dxmmu2 * self.xmmu
        dxmmu += dxnorm * (1 / self.std)
        dmu = -np.sum(dxmmu, axis=0)
        dx = dxmmu
        dx += dmu * np.ones((self.N, self.D)) * (1 / self.N)
        return dx


class FastBatchNorm(Module):
    """
    Implements BatchNorm layer.
    """

    def __init__(self, num_features):
        """
        Args:
            num_features (int): D from an input of size (N, D)
        """
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-8
        # initialize gamma to ones
        self.gamma = Parameter(np.ones(num_features))
        # initialize beta to zeros
        self.beta = Parameter(np.zeros(num_features))
        # append parameters to _parameters
        self._parameters['gamma'] = self.gamma
        self._parameters['beta'] = self.beta

    def forward(self, X):
        """
        Implements forward pass of the BatchNorm layer.
        """
        self.N, self.D = X.shape
        if self.D != self.num_features:
            raise ValueError(
                "Mismatch in number of features between X and batchnorm layer")
        self.x = X
        # calculate the mean of every feature
        self.mu = np.mean(self.x, axis=0)
        # calculate the variance of every feature
        self.var = np.var(self.x, axis=0)
        # calculate the std deviation of every feature
        # add eps to prevent divide by zero while calculating (x - mu)/std
        self.std = np.sqrt(self.var + self.eps)
        # calculate x-mu
        self.xmmu = self.x - self.mu
        # calculate x_norm
        self.x_norm = self.xmmu / self.std
        # shift and scale x_norm by gamma and beta
        out = self.gamma.data * self.x_norm + self.beta.data
        return out

    def backward(self, dout):
        """
        Implements backward pass of the BatchNorm layer.

        Args:
            dout: gradient from the upstream.
        """
        pass
