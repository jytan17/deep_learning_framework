# TODO: Add conv layer

from pydeep.core import Function, Variable
import numpy as np

class Layer(Function):

    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.name = name
        self.weights = {}

    def _init_weights(self, *args, **kwargs):
        pass

class Linear(Layer):

    def __init__(self, in_dim, out_dim):
        name = f"Linear: ({in_dim}, {out_dim})"
        super().__init__(name)
        self._init_weights(in_dim, out_dim)

    def _init_weights(self, in_dim, out_dim):
        scale = 1 / np.sqrt(in_dim)
        w = scale * np.random.rand(in_dim, out_dim)
        b = scale * np.random.rand(1, out_dim)
        self.weights['w'] = Variable(data = w)
        self.weights['b'] = Variable(data = w)

    def forward(self, x):
        w = self.weights['w'].data
        b = self.weights['b'].data

        y = np.dot(x, w + b)

        self.cache['input'] = x
        self.cache['output'] = y

        return y

    def local_grad(self, x):
        self.cache['local_gx'] = self.weights['w'].data
        self.cache['local_gw'] = x
        self.cache['local_gb'] = np.ones_like(self.weights['b'].data)

    def backward(self, gy):
        gx = gy.dot(self.cache['local_gx'].T) # gradient for backprop
        gw = self.cache['local_gw'].T.dot(gy) # local gradient of weights w
        gb = np.sum(gy, axis = 0, keepdims=True) # local gradient of weights b

        self.weights['w'].grad = gw # store the gradients for update
        self.weights['b'].grad = gb # store the gradients of

        return gx


