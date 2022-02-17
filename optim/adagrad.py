from cupy import sqrt
from .optimizer import Optimizer

class Adagrad(Optimizer):
    """
    Adagrad optmizer.
    """

    def __init__(self, parameters, lr, reg=0):
        """
        Args:
            parameters: model parameters.
            lr: learning rate.
            reg: regularization parameter.
        """
        super().__init__(parameters)
        self.lr = lr
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            if self.reg != 0:
                param.grad += self.reg * param.data
            param.s += param.grad ** 2
            self._step(param.data, param.grad, param.s)

    def _step(self, data, grad, s):
        """
        Updates model parameter.
        """
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * grad) / sqrt(s)
            
