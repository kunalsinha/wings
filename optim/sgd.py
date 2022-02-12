from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer implementation.
    """

    def __init__(self, params, lr, momentum=0, reg=0):
        """
        SGD optimizer with momentum

        Args:
            lr (float): learning rate
            momentum (float): hyperparameter for SGD with momentum
            reg (float): regularization hyperparameter
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.reg = reg

    def step(self):
        """
        Step through each model parameter and update.
        Weight decay gradient is computed as
            reg * weight
        """
        for param in self.parameters:
            grad = param.grad
            # calculate grad contribution from weight decay
            if self.reg != 0:
                grad += self.reg * param.data
            self._step(param.data, grad)

    def _step(self, data, grad):
        data -= self.lr * grad
