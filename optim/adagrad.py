from numpy import sqrt
from .optimizer import Optimizer

class Adagrad(Optimizer):

    def __init__(self, parameters, lr, reg=0):
        super().__init__(parameters)
        self.lr = lr
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        for param in self.parameters:
            param.s += param.grad ** 2
            self._step(param.data, param.grad, param.s)

    def _step(self, data, grad, s):
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * grad) / sqrt(s)
            
