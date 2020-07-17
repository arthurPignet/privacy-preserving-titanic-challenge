import logging

import numpy as np


class LogisticRegression:
    """
        Model of logistic regression, performed on unencrypted data.
        It aim to be a reference model, to compare with encrypted models

    """

    def __init__(self,
                 init_weight,
                 init_bias,
                 lr=1,
                 max_epoch=100,
                 reg_para=0.5,
                 verbose=-1,
                 save_weight=-1,
                 ):
        """

            Constructor


            :param init_weight : Initial weight
            :param init_bias : Initial weight
            :param lr: float. learning rate
            :param max_epoch: int. number of epoch to be performed
            :param reg_para: float. regularization parameter
            :param verbose: int. number of epoch were the loss is not computed, nor printed.
                            Every <verbose> epoch, the loss will be logged in loss_list.
                            If set to -1, the loss will not be computed nor stored at all

            :param save_weight: int. number of epoch were the weight will be stored.
                                Every <save_weight> epoch, the weight will be logged in weight_list
                                If set to -1, the weight will not be saved
            """

        self.logger = logging.getLogger(__name__)

        self.verbose = verbose
        self.save_weight = save_weight

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = lr
        if save_weight > 0:
            self.weight_list = []
            self.bias_list = []
        if verbose > 0:
            self.loss_list = []
        self.weight = init_weight
        self.bias = init_bias

    def loss(self, X, Y):
        """
            This method compute the cross entropy loss.
            :param X: samples
            :param Y: labels
            :return: loss. float

        """
        re = X.dot(self.weight) + self.bias  # we use cross entropy loss function
        prediction = (np.float_power(re, 3)) * -0.004 + re * 0.197 + 0.5
        loss = -np.log(prediction).dot(Y)
        loss -= (1 - np.array(Y)).T.dot(np.log(1 - prediction))
        loss += (self.reg_para / 2) * (np.array(self.weight).dot(self.weight) + np.float_power(self.bias, 2))
        return loss

    def accuracy(self, X, Y):
        """
            This method compute the accuracy
            :param X: samples of the data on which the accuracy will be computed
            :param Y: labels of the data on which the accuracy will be computed
            :return: accuracy
        """
        prediction = self.predict(X)
        return (np.abs((Y - prediction.reshape(Y.shape))) < 0.5).astype(float).mean()

    @staticmethod
    def sigmoid(enc_x, mult_coeff=1):
        """
            Sigmoid implementation
            We use the polynomial approximation of degree 3
            sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
            from https://eprint.iacr.org/2018/462.pdf

            :param enc_x:
            :param mult_coeff: The return is equivalent to sigmoid(x) * mult_coeff, but save one homomorph multiplication
            :return: CKKS vector (result of sigmoid(x))
        """
        return (np.power(enc_x, 3) * -0.004 + enc_x * 0.197 + 0.5) * mult_coeff

    def forward(self, vec, mult_coeff=1):
        """
            Compute forward propagation on plaintext (or a list of plaintext)
            :param vec: plaintext or list of plaintext on which we want to make predictions (ie forward propagation
            :param mult_coeff: the result will be
            :return: prediction or list of predictions
        """
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegression.sigmoid(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(self.weight)
            return LogisticRegression.sigmoid(res, mult_coeff=mult_coeff)

    def backward(self, X, predictions, Y):
        """
            Compute the backpropagation on a given batch
            :param X: list of vectors. Features of the data on which the gradient will be computed (backpropagation)
            :param predictions: np.array of prediction. Label predictions (forward propagation) on the data on which the gradient will be computed (backpropagation)
            :param Y: np.array of label. Label of the data on which the gradient will be computed (backpropagation)
            :return: np.array : gradient
        """
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

    def fit(self, X, Y):
        """
        Train the model over encrypted data.
        Unencrypted data can be provided, this is not safe, but can be useful for debug and monitoring

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)
        """
        while self.iter < self.num_iter:

            prediction = self.forward(X)  # we can add batching later
            direction_weight, direction_bias = self.backward(X, prediction, Y)
            self.bias -= direction_bias
            self.weight -= direction_weight

            if self.verbose > 0 and self.iter % self.verbose == 0:
                self.logger.info("iteration number %d is starting" % (self.iter + 1))
                self.loss_list.append(self.loss(X, Y))
                self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
            if self.save_weight > 0 and self.iter % self.save_weight == 0:
                self.weight_list.append(self.weight)
                self.bias_list.append(self.bias)

            self.iter += 1

    def predict(self, X):
        """
            Use the model to predict a label.
            :param X: encrypted CKKS vector
            :return: encrypted prediction
        """
        return self.forward(X)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
