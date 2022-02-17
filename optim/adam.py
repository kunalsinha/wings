from numpy import sqrt
from .optimizer import Optimizer

class Adam(Optimizer):
    """
    Implements Adam optimizer.
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.99), reg=0):
        """
        Args:
            parameters: model parameters
            lr: learning rate
            betas: coefficients used for computing running averages of the
                gradient and its square in that order
            reg: regularization parameter for weight decay
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            param.v = self.beta1 * param.v + (1 - self.beta1) * param.grad
            param.s = self.beta2 * param.s + (1 - self.beta2) * (param.grad ** 2)
            self._step(param.data, param.v, param.s)

    def _step(self, data, v, s):
        """
        Updates the model parameter.
        """
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * v) / sqrt(s)
            


