import numpy as np

class Variable:

    def __init__(self, data):
        self.data = data if not isinstance(data, np.ndarray) else np.array(data)
        self.grad = None

    def zero_grad(self):
        self.grad = None


class Function:

    def __init__(self, *args, **kwargs):
        self.cache = {}

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        self.local_grad(*args, **kwargs)
        return output

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def local_grad(self, *args, **kwargs):
        pass