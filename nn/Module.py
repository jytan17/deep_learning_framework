from pydeep.layer import Layer


class Sequential:

    def __init__(self, layers, loss, optim):
        self.layers = layers
        self.loss_fn = loss
        self.optim = optim

    def __call__(self, x, training = True):
        out = self.forward(x)
        if not training:
            self.zero_grad()

        return out

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def loss(self, x, y):
        return self.loss_fn(x, y)

    def backward(self):
        gx = self.loss_fn.backward()
        for l in reversed(self.layers):
            gx = l.backward(gx)

    def step(self):
        self.optim.step(self.layers)

    def clear_cache(self):
        for l in self.layers:
            l.cache = {}

    def zero_grad(self):
        for l in self.layers:
            l.cache = {}
            if isinstance(l, Layer):
                l.weights['w'].grad = None
                l.weights['b'].grad = None

    def __repr__(self):
        repr = "Sequential"
        for l in self.layers:
            repr += f"\n{l.name}"



    # @property
    # def parameters(self):
    #     model_params = {}
    #     for l in self.layers:
    #         if isinstance(l, Layer):
    #             model_params[l.name] = {'w': l.weights['w'], 'b': l.weights['b']}
    #
    #     return model_params
    #
    # @parameters.setter
    # def parameters(self, updates):
    #     for l in self.layers:
    #         if isinstance(l, Layer):
    #             name = l.name
    #             l.weights['w'].data = updates[name]['w']
    #             l.weights['b'].data = updates[name]['b']





