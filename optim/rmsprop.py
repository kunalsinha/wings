from numpy import sqrt
from .optimizer import Optimizer

class RMSprop(Optimizer):

    def __init__(self, parameters, lr, beta=0.9, reg=0):
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        for param in self.parameters:
            param.s = self.beta * param.s + (1 - self.beta) * (param.grad ** 2)
            self._step(param.data, param.grad, param.s)

    def _step(self, data, grad, s):
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * grad) / sqrt(s)
            

