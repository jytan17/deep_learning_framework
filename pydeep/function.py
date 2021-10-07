from pydeep.core import Function
import pydeep.functional as F

class Sigmoid(Function):

    def forward(self, x):
        return F.sigmoid(x)

    def local_grad(self, x):
        gx = F.sigmoid_prime(x)
        self.cache["local_gx"] = gx

    def backward(self, gy):
        return gy * self.cache["local_gx"]

class ReLU(Function):

    def forward(self, x):
        return F.relu(x)

    def local_grad(self, x):
        gx = F.relu_prime(x)
        self.cache["local_gx"] = gx

    def backward(self, gy):
        return gy * self.cache["local_gx"]

class LeakyReLU(Function):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leaky_relu(x, self.alpha)

    def local_grad(self, x):
        gx = F.leaky_relu_prime(x, self.alpha)
        self.cache["local_gx"] = gx

    def backward(self, gy):
        return gy * self.cache["local_gx"]