from .module import Module
import cupy as np

class Sequential(Module):
    """
    Stores modules sequentially that allows for an easy
    way of building reusable blocks of layers.
    """

    def __init__(self, *modules):
        """
        Create a sequence of passed modules.

        Args:
            modules: list of Module type layers.
        """
        super().__init__()
        for idx, m in enumerate(modules):
            self._modules[str(idx)] = m

    def forward(self, X):
        """
        Forward propagate through all the layers in the
        block sequentially.

        Args:
            X: input
        """
        for name, layer in self._modules.items():
            X = layer(X)
        return X

    def backward(self, dout):
        """
        Backpropagate through the layers in reverse order.

        Args:
            dout: gradient from above.
        """
        for name, layer in reversed(self._modules.items()):
            dout = layer.backward(dout)
        return dout

    def add_module(self, module):
        """
        Add a new layer to the block.

        Args:
            m_name (str): name of the new module
            module: module to be added
        """
        key = int(list(self._modules.keys())[-1]) + 1
        self._modules[str(key)] = module

    def __call__(self, X):
        return self.forward(X)


