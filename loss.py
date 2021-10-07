from pydeep.core import Function
import numpy as np

class Loss(Function):

    def backward(self):

        gx = self.cache["local_gx"]
        return gx

class CrossEntropyLoss(Loss):

    def forward(self, x, y):
        """
        Descr.: calculate the mean loss of the batch

        :param x:   final layer of the network (batch_size, n_dim),
                    no need for softmax as we calculate it here
        :param y:   truth values (batch_size, 1), each entry in the
                    array is the an integer indicating the class
                    of the data point
        :return:    the mean cross entropy loss of the batch
        """

        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        logprobs = -np.log([probs[i, y[i]] for i in range(len(probs))])
        crossentropy_loss = np.mean(logprobs)

        self.cache["probs"] = probs
        self.cache['y'] = y

        return crossentropy_loss

    def local_grad(self, x, y):
        """
        Descr.:     cache the local gradient for backprop.

        :param x:   final layer of the network (batch_size, n_dim),
                    no need for softmax as we calculate it here
        :param y:   truth values (batch_size, 1), each entry in the
                    array is the an integer indicating the class
                    of the data point
        """

        probs = self.cache["probs"]
        truth = np.zeros_like(probs)
        for i, j in enumerate(y):
            truth[i, j] = 1.0

        self.cache["local_gx"] = (probs - truth) / float(len(x))



class MSELoss(Loss):

    def forward(self, x, y):
        """
        Descr.:     calculate mean loss of the batch, the avg. loss of the current batch

        :param x:   predicted values, (batch_size, 1)
        :param y:   truth values, (batch_sze, 1)
        :return:    scalar value, the mse of the batch
        """
        loss = ((x - y)**2).mean()
        return loss

    def local_grad(self, x, y):
        """
        Descr.: calulate dL/dx, and cache it for backprop.

        :param x: predicted values, (batch_size, 1)
        :param y: truth values, (batch_sze, 1)
        """
        self.cache["local_gx"] = 2 * (x - y) / x.shape[0]
