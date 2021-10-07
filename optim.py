from pydeep.layer import Layer


class SGD:

    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):
        for l in layers:
            if isinstance(l, Layer):
                l.weights['w'].data -= self.lr * l.weights['w'].grad
                l.weights['b'].data -= self.lr * l.weights['b'].grad

