import logging

import numpy as np


class LogisticRegressionHE:
    def __init__(self,
                 init_weight,
                 init_bias,
                 weight_ne,
                 bias_ne,
                 refresh_function,
                 confidential_kwarg,
                 loss=None,
                 accuracy=None,
                 lr=1,
                 num_iter=100,
                 reg_para=0.5,
                 verbose=-1,
                 safety=False,
                 ):
        #TODO doc
        self.logger = logging.getLogger(__name__)

        self.refresh_function = refresh_function
        self.confidential_kwarg = confidential_kwarg
        self.loss_function = loss
        self.accuracy_function = accuracy

        self.verbose = verbose
        self.safety = safety

        self.iter = 0
        self.num_iter = num_iter
        self.reg_para = reg_para
        self.lr = lr

        if not safety:
            self.direction_ne = []
        self.loss_list = []
        self.grad_norm = []
        self.weight_ne = weight_ne
        self.weight = init_weight
        self.bias_ne = bias_ne
        self.bias = init_bias

    def refresh(self, vector):
        return self.refresh_function(vector, **self.confidential_kwarg)

    def loss(self):
        return self.loss_function(self.weight, self.bias, **self.confidential_kwarg)

    def accuracy(self, unencrypted_X=None, unencrypted_Y=None):
        return self.accuracy_function(self.weight, self.bias, unencrypted_X, unencrypted_Y, **self.confidential_kwarg)

    @staticmethod
    def sigmoid(enc_x, mult_coeff=1):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        poly_coeff = [0.5, 0.197, 0, -0.004]
        return enc_x.polyval([i * mult_coeff for i in
                              poly_coeff])  # The use of mult_coeff allows us to multiply the encrypted result of the polynomial evaluation without homomorphique multiplication

    @staticmethod
    def sigmoid_ne(enc_x, mult_coeff=1):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        poly_coeff = [0.5, 0.197, 0, -0.004]
        return (np.power(enc_x, 3) * -0.004 + enc_x * 0.197 + 0.5) * mult_coeff

    def forward(self, vec, mult_coeff=1):
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(self.weight) + self.bias
            return LogisticRegressionHE.sigmoid(res, mult_coeff=mult_coeff)

    def forward_ne(self, vec, mult_coeff=1):
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid_ne(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(self.weight_ne)
            return LogisticRegressionHE.sigmoid_ne(res, mult_coeff=mult_coeff)

    def backward(self, X, predictions, Y):
        inv_n = 1. / len(Y)
        err = predictions[0] - Y[0]
        direction_weight = X[0] * err
        direction_bias = err
        for i in range(1, len(X)):
            err = predictions[i] - Y[i]
            direction_weight += X[i] * err
            direction_bias += err
        direction_weight = (direction_weight * (inv_n * self.lr)) + (self.weight * (inv_n * self.lr * self.reg_para))
        direction_bias = direction_bias * (inv_n * self.lr)
        return direction_weight, direction_bias

    def backward_ne(self, X, predictions, Y):
        inv_n = 1. / len(Y)
        err = predictions[0] - Y[0]
        direction_weight = X[0] * err
        direction_bias = err
        for i in range(1, len(X)):
            err = predictions[i] - Y[i]
            direction_weight += X[i] * err
            direction_bias += err
        direction_weight = (direction_weight * (inv_n * self.lr)) + (self.weight_ne * (inv_n * self.lr * self.reg_para))
        direction_bias = direction_bias * (inv_n * self.lr)
        return direction_weight, direction_bias

    def fit(self, X, Y, X_ne=None, Y_ne=None):

        while self.iter < self.num_iter:

            self.weight = self.refresh(self.weight)
            self.bias = self.refresh(
                self.bias)  # refreshing the init_weight and the init_bias to avoid scale out of bound
            encrypted_prediction = self.forward(X)  # we can add batching later
            direction_weight, direction_bias = self.backward(X, encrypted_prediction, Y)
            prediction = self.forward_ne(X_ne)
            ne_direction_weight, ne_direction_bias = self.backward_ne(X_ne, prediction, Y_ne)
            self.weight -= direction_weight
            self.weight_ne -= ne_direction_weight
            self.bias -= direction_bias
            self.bias_ne -= ne_direction_bias
            self.iter += 1
            if self.verbose > 0 and self.iter % self.verbose == 0:
                self.logger.info("iteration number %d is starting" % self.iter)
                self.loss_list.append(self.loss())
                self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
                if not self.safety:
                    prediction = self.forward_ne(X_ne)
                    ne_direction_weight, ne_direction_bias = self.backward_ne(X_ne, prediction, Y_ne)
                    self.direction_ne.append((ne_direction_weight, ne_direction_bias))
                    self.grad_norm.append((direction_weight.decrypt(), direction_bias.decrypt()))
                    self.logger.debug(
                        "error %d" % (np.sum(
                            np.power((np.array(direction_weight.decrypt()) - ne_direction_weight), 2)) / np.sum(
                            np.power(ne_direction_weight, 2))))

    def predict(self, X):
        return self.forward(X)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
