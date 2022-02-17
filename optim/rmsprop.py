from cupy import sqrt
from .optimizer import Optimizer

class RMSprop(Optimizer):
    """
    RMSprop optimization algorithm.
    """

    def __init__(self, parameters, lr, beta=0.9, reg=0):
        """
        Args:
            parameters: model parameters.
            lr: learning rate.
            beta: coefficient used to compute running avg of squared gradients.
            reg: regularization parameter for weight decay.
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            if self.reg != 0:
                param.grad += self.reg * param.data
            param.s = self.beta * param.s + (1 - self.beta) * (param.grad ** 2)
            self._step(param.data, param.grad, param.s)

    def _step(self, data, grad, s):
        """
        Updates model parameter.
        """
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * grad) / sqrt(s)
            

