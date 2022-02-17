from numpy import sqrt

class Adadelta:

    def __init__(self, parameters, lr=1.0, rho=0.9, eps=1e-8, reg=0):
        self.parameters = parameters
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.reg = reg

    def step(self):
        for param in self.parameters:
            param.s = self.rho * param.s + (1 - self.rho) * (param.grad ** 2)
            delta = ((sqrt(param.v + self.eps) / sqrt(param.s + self.eps)) *
                        param.grad)
            param.v = self.rho * param.v + (1 - self.rho) * (delta ** 2)
            self._step(param.data, delta)

    def _step(self, data, grad):
        data -= self.lr * grad

