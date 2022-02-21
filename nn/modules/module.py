import wings


class Module:
    """
    Base class for all layers.
    """

    def __init__(self) -> None:
        self._modules = {}  # stores all child modules
        self._parameters = {}  # stores self parameters
        self._mode = "train"  # train or eval mode

    def __setattr__(self, name, value):
        """Add all module attributes to _modules list."""
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def apply(self, fn):
        """
        Apply the given function to all the modules in _modules list as well
        as self.

            Args:
                fn (function): function to be applied
        """
        for module in self._modules.values():
            fn(module)
        fn(self)

    def set_mode(self, mode):
        """
        Set model to the requested mode

        Args:
            mode (str): 'train' or 'eval'
        """
        if mode not in ["train", "eval"]:
            raise ValueError("Invalid mode argument")
        # set mode for all child modules
        for module in self._modules.values():
            if type(module) == wings.nn.Sequential:
                module.set_mode(mode)
            module._mode = mode
        # set self mode
        self._mode = mode

    def state_dict(self):
        """
        Returns a dictionary containing all model parameters.
        """
        state = {}
        # add self parameters to dictionary
        for p_name, param in self._parameters.items():
            state[p_name] = param.data
        # add child module parameters to dictionary
        for m_name, module in self._modules.items():
            for p_name, param in module._parameters.items():
                state[f"{m_name}.{p_name}"] = param.data
        return state

    def parameters(self):
        """
        Returns a list of all model parameters.
        """
        params = []
        # add self parameters to the list
        for parameter in self._parameters.values():
            params.append(parameter)
        # add child module parameters to the list
        for module in self._modules.values():
            if type(module) == wings.nn.Sequential:
                params += module.parameters()
            else:
                for parameter in module._parameters.values():
                    params.append(parameter)
        return params

    def zero_grad(self):
        """
        Set the gradient to zero for all model parameters.
        """
        # set gradient of self parameters to zero
        for parameter in self._parameters.values():
            parameter.zero_grad()
        # set gradient of child module parameters to zero
        for module in self._modules.values():
            for parameter in module._parameters.values():
                parameter.zero_grad()

    def __call__(self, X):
        return self.forward(X)

    def __getitem__(self, key):
        return self._modules[key]

    def __repr__(self):
        r = f"{self.__class__.__name__}(\n"
        for m_name, module in self._modules.items():
            r += f"  ({m_name}): {repr(module)}\n"
        r += f")"
        return r
