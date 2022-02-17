from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters, lr, momentum=0, reg=0):
        """
        Args:
            parameters: model parameters
            lr: learning rate
            momentum: coefficient to compute the running avg of gradient
            reg: regularization parameter
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.reg = reg

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            grad = param.grad
            # calculate grad contribution from weight decay
            if self.reg != 0:
                grad += self.reg * param.data
            if self.momentum != 0:
                param.v = self.momentum * param.v + (1 - self.momentum) * grad
                self._step(param.data, param.v)
            else:
                self._step(param.data, grad)

    def _step(self, data, grad):
        """
        Updates model parameter.
        """
        data -= self.lr * grad
