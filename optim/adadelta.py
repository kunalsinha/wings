from cupy import sqrt
from .optimizer import Optimizer

class Adadelta(Optimizer):
    """
    Adadelta optimization algorithm.
    """

    def __init__(self, parameters, lr=1.0, rho=0.9, eps=1e-8, reg=0):
        """
        Args:
            parameters: model parameters.
            lr: coefficient to scale the delta before adding to the parameter.
            rho: coefficient to compute the running average of squared gradients.
            eps: quantity added to improve numerical stability.
            reg: regularization constant for weight decay.
        """
        super().__init__(parameters)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.reg = reg

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            if self.reg != 0:
                param.grad += self.reg * param.data
            param.s = self.rho * param.s + (1 - self.rho) * (param.grad ** 2)
            delta = ((sqrt(param.v + self.eps) / sqrt(param.s + self.eps)) *
                        param.grad)
            param.v = self.rho * param.v + (1 - self.rho) * (delta ** 2)
            self._step(param.data, delta)

    def _step(self, data, grad):
        """
        Updates model parameter.
        """
        data -= self.lr * grad

