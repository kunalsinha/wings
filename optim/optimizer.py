import cupy as np

class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        """
        Set grad for all parameters to zero.
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    def l2_loss(self):
        """
        Calculates the l2 loss of the model.
        """
        sum = 0.0
        for p in self.parameters:
            sum += (self.reg / 2) * np.sum(p.data ** 2)
        return sum
    

