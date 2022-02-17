from numpy import sqrt
from .optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.99, reg=0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.reg = reg
        self.eps = 1e-8

    def step(self):
        for param in self.parameters:
            param.v = self.beta1 * param.v + (1 - self.beta1) * param.grad
            param.s = self.beta2 * param.s + (1 - self.beta2) * (param.grad ** 2)
            self._step(param.data, param.v, param.s)

    def _step(self, data, v, s):
        # add eps to prevent divide by zero error
        s += self.eps
        data -= (self.lr * v) / sqrt(s)
            


