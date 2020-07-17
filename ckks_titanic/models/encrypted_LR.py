import logging

import numpy as np


class LogisticRegressionHE:
    """
        Model of logistic regression, performed on encrypted data, using homomorphic encryption, especially the CKKS scheme implemented in tenSEAL

    """

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
                 max_epoch=100,
                 reg_para=0.5,
                 verbose=-1,
                 safety=False,
                 ):
        """

            Constructor

            :param init_weight: CKKS vector. Initial weight
            :param init_bias: CKKS vector. Initial weight
            :param (Optional) weight_ne: numpy array. Initial unencrypted weight : Needed when safety is set to false, to compute the relative error between the homomorph gradient and the classic gradient
            :param (Optional) bias_ne: numpy array. Initial unencrypted bias. Needed when safety is set to false, to compute the relative error between the homomorph gradient and the classic gradient
            :param refresh_function: function. Refresh ciphertext
            :param confidential_kwarg: dict. Will be passed as **kwarg to refresh, loss and accuracy functions. Contain confidential data which are needed by those functions.
            :param loss: function. Compute cross entropy loss
            :param accuracy: function. Compute accuracy
            :param lr: float. learning rate
            :param max_epoch: int. number of epoch to be performed
            :param reg_para: float. regularization parameter
            :param verbose: int. number of epoch were the loss (and the error if safety is set to False) is not computed, nor printed. Every <verbose> epoch, the loss (and error) will be logged
            :param safety: boolean. If True, the protocol is as secure as refresh, loss and accuracy functions are. The unencrypted data are not necessary, as the error will not be computed.


        """
        self.logger = logging.getLogger(__name__)

        self.refresh_function = refresh_function
        self.confidential_kwarg = confidential_kwarg
        self.loss_function = loss
        self.accuracy_function = accuracy

        self.verbose = verbose
        self.safety = safety

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = lr

        if not safety:
            self.error = []
        self.loss_list = []
        self.weight_ne = weight_ne
        self.weight = init_weight
        self.bias_ne = bias_ne
        self.bias = init_bias

    def refresh(self, vector):
        """
            The method refresh the depth of a ciphertext. It call the refresh function which aims to refresh ciphertext by preserving privacy
            :param vector: CKKS vector, ciphertext
            :return: refreshed CKKS vector
        """
        return self.refresh_function(vector, **self.confidential_kwarg)

    def loss(self):
        """
            This method compute the loss by getting it from the loss_function, which aims to compute loss by preserving private.
            :return:
            loss : float
        """
        return self.loss_function(self.weight, self.bias, self.reg_para, **self.confidential_kwarg)

    def accuracy(self, unencrypted_X=None, unencrypted_Y=None):
        """
            This method compute the accuracy by getting it from the accuracy_function, which aims to compute accuracy by preserving
            :param unencrypted_X: samples of the data on which the accuracy will be computed
            :param unencrypted_Y: labels of the data on which the accuracy will be computed
            :return: accuracy
        """
        return self.accuracy_function(self.weight, self.bias, prior_unencrypted_X=unencrypted_X,
                                      prior_unencrypted_Y=unencrypted_Y, **self.confidential_kwarg)

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
        poly_coeff = [0.5, 0.197, 0, -0.004]
        return enc_x.polyval([i * mult_coeff for i in poly_coeff])

    @staticmethod
    def sigmoid_ne(enc_x, mult_coeff=1):
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
            Compute forward propagation on a CKKS vector (or a list of CKKS vectors)
            :param vec: CKKS vector or list of CKKS vector on which we want to make predictions (ie forward propagation
            :param mult_coeff: The return is equivalent to forward(x) * mult_coeff, but save one homomorph multiplication
            :return: encrypted prediction or list of encrypted predictions
        """
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(self.weight) + self.bias
            return LogisticRegressionHE.sigmoid(res, mult_coeff=mult_coeff)

    def forward_ne(self, vec, mult_coeff=1):
        """
            Compute forward propagation on plaintext (or a list of plaintext)
            :param vec: plaintext or list of plaintext on which we want to make predictions (ie forward propagation
            :param mult_coeff: the result will be
            :return: prediction or list of predictions
        """
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid_ne(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(self.weight_ne)
            return LogisticRegressionHE.sigmoid_ne(res, mult_coeff=mult_coeff)

    def backward(self, X, predictions, Y):
        """
            Compute the backpropagation on a given encrypted batch
            :param X: list of encrypted (CKKS vectors). Features of the data on which the gradient will be computed (backpropagation)
            :param predictions: list of encrypted CKKS vectors. Label predictions (forward propagation) on the data on which the gradient will be computed (backpropagation)
            :param Y: list of encrypted CKKS vectors. Label of the data on which the gradient will be computed (backpropagation)
            :return: CKKS vector. Encrypted gradient
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

    def backward_ne(self, X, predictions, Y):
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
        direction_weight = (direction_weight * (inv_n * self.lr)) + (self.weight_ne * (inv_n * self.lr * self.reg_para))
        direction_bias = direction_bias * (inv_n * self.lr)
        return direction_weight, direction_bias

    def fit(self, X, Y, X_ne=None, Y_ne=None):
        """
        Train the model over encrypted data.
        Unencrypted data can be provided, this is not safe, but can be useful for debug and monitoring

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)
        :param (Optional) X_ne: np.array: samples (train set)
        :param (Optional) Y_ne: np.array: labels (train set)
        """
        while self.iter < self.num_iter:

            self.weight = self.refresh(self.weight)
            self.bias = self.refresh(
                self.bias)  # refreshing the init_weight and the init_bias to avoid scale out of bound
            # encrypted gradient descent
            encrypted_prediction = self.forward(X)  # we can add batching later
            direction_weight, direction_bias = self.backward(X, encrypted_prediction, Y)
            self.bias -= direction_bias
            self.weight -= direction_weight
            # unencrypted gradient descent
            if self.verbose > 0:
                unencrypted_prediction = self.forward_ne(X_ne)
                ne_direction_weight, ne_direction_bias = self.backward_ne(X_ne, unencrypted_prediction, Y_ne)
                self.weight_ne -= ne_direction_weight
                self.bias_ne -= ne_direction_bias
                if self.iter % self.verbose == 0:
                    self.logger.info("iteration number %d is starting" % (self.iter + 1))
                    self.loss_list.append(self.loss())
                    self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
                    if not self.safety:
                        err = (np.sum(
                            np.power((np.array(direction_weight.decrypt()) - ne_direction_weight), 2)) /
                               np.sum(np.power(ne_direction_weight, 2)))
                        self.error.append(err)
                        self.logger.info(
                            "error %d" % self.error[-1])

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
