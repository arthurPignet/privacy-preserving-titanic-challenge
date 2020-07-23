import logging
import multiprocessing

import numpy as np


class LogisticRegression:
    """
        Model of logistic regression, performed on unencrypted data.
        It aim to be a reference model, to compare with encrypted models

    """

    def __init__(self,
                 init_weight,
                 init_bias,
                 learning_rate=1,
                 momentum_rate=0,
                 max_epoch=100,
                 reg_para=0.5,
                 n_jobs=None,
                 verbose=-1,
                 save_weight=-1
                 ):
        """

            Constructor



            :param init_weight : Initial weight
            :param init_bias : Initial weight
            :param learning_rate: float. see Nesterov Accelerated Gradient Optimizer
            :param momentum_rate: float. see Nesterov Accelerated Gradient Optimizer
            :param max_epoch: int. number of epoch to be performed
            :param reg_para: float. regularization parameter
            :param verbose: int. number of epoch were the loss is not computed, nor printed.
                            Every <verbose> epoch, the loss will be logged in loss_list.
                            If set to -1, the loss will not be computed nor stored at all

            :param save_weight: int. number of epoch were the weight will be stored.
                                Every <save_weight> epoch, the weight will be logged in weight_list
                                If set to -1, the weight will not be saved
            """


        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)

        self.verbose = verbose
        self.save_weight = save_weight

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = learning_rate
        self.mr = momentum_rate

        if save_weight > 0:
            self.weight_list = []
            self.bias_list = []
        if verbose > 0:
            self.loss_list = []
            self.true_loss_list = []
        self.weight = init_weight.copy()
        self.bias = init_bias.copy()

    def refresh(self, x):
        self.iter += 0
        return x

    @staticmethod
    def _log(x, mult_coeff=1):
        poly_coeff = [-3.69404813, 13.30907268, -19.06853265, 9.63445963]
        res = 0
        for i in range(len(poly_coeff)):
            res += poly_coeff[i] * np.power(x, i) * mult_coeff
        return res

    def loss(self, predictions, Y):
        """
            This method compute the cross entropy loss.
            :param predictions: precomputed predictions
            :param Y: labels
            :return: loss. float

        """
        predictions = np.array(predictions).reshape(Y.shape)
        loss = -self._log(predictions).T.dot(Y)
        loss = loss - (1 - np.array(Y)).T.dot(self._log(1 - np.array(predictions)))
        loss = loss + ((self.reg_para / 2) * (np.array(self.weight).dot(self.weight) + np.float_power(self.bias, 2)))
        return loss.reshape(1)[0]/len(Y)

    def true_loss(self, X, Y):
        predictions = 1 / (1 + np.exp(-(X.dot(self.weight) + self.bias)))
        loss = -np.log(predictions).T.dot(Y)
        loss = loss - (1 - np.array(Y)).T.dot(np.log(1 - np.array(predictions)))
        loss = loss + ((self.reg_para / 2) * (np.array(self.weight).dot(self.weight) + np.float_power(self.bias, 2)))
        return loss[0]/len(Y)

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
            res = vec.dot(self.weight) + self.bias
            return LogisticRegression.sigmoid(res, mult_coeff=mult_coeff)

    @staticmethod
    def backward(X, predictions, Y):
        """
            Compute the backpropagation on a given batch
            :param X: list of vectors. Features of the data on which the gradient will be computed (backpropagation)
            :param predictions: np.array of prediction. Label predictions (forward propagation) on the data on which the gradient will be computed (backpropagation)
            :param Y: np.array of label. Label of the data on which the gradient will be computed (backpropagation)
            :return: np.array : gradient
        """
        if type(X) == list:
            err = predictions[0] - Y[0]
            print('LIST')
            grad_weight = X[0] * err
            grad_bias = err
            for i in range(1, len(X)):
                err = predictions[i] - Y[i]
                grad_weight += X[i] * err
                grad_bias += err
            return grad_weight, grad_bias
        else:
            err = predictions - Y
            grad_weight = X * err
            grad_bias = err

            return grad_weight, grad_bias

    def _forward_backward_wrapper(self, arg):
        """
        Wrapper for forward_backward, which expands the parameter tuple to forward_backward
        :param arg: Tuple, (X,Y) with X standing for features of the data on which predictions will be made, (forward propagation) and then the gradient will be computed (backpropagation)
                                  and Y standing for label of the data on which the gradient will be computed (backpropagation)
        :return:
                Tuple with 3 vectors.  batch_gradient for weight and bias, and batch predictions.
        """
        return self.forward_backward(*arg)

    def forward_backward(self, X, Y):
        """
        Perform forward propagation, and then backward propagation.
        :param X:  Features of the data on which predictions will be made, (forward propagation) and then the gradient will be computed (backpropagation)
        :param Y: . Label of the data on which the gradient will be computed (backpropagation)
        :return: : Tuple with 3  vectors.  batch_gradient for weight and bias, and batch predictions.

        """
        prediction = self.forward(X)
        grads = self.backward(X, prediction, Y)
        return grads[0], grads[1], prediction

    def fit(self, X, Y):
        """
        Train the model over encrypted data.
        Unencrypted data can be provided, this is not safe, but can be useful for debug and monitoring

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)
        """
        batches = [(x, y) for x, y in zip(X, Y)]
        inv_n = (1 / len(Y))

        nag_weight = self.weight
        nag_bias = self.bias

        while self.iter < self.num_iter:

            nag_weight = self.refresh(nag_weight)
            nag_bias = self.refresh(nag_bias)
            
            prev_weight = self.weight.copy()
            prev_bias = self.bias.copy()
            self.weight = nag_weight
            self.bias = nag_bias

            if self.n_jobs > 1:
                try:
                    process = multiprocessing.Pool(
                        processes=self.n_jobs)  # can be done while waiting for the refreshed weight
                    multiprocess_results = process.map_async(self._forward_backward_wrapper, batches)
                    process.close()
                    process.join()
                    directions = multiprocess_results.get()
                except:
                    self.logger.warning("One tenseal object cannot be pickle, aborting the use of multiprocessing.")
                    directions = [self._forward_backward_wrapper(i) for i in batches]
            else:
                directions = [self._forward_backward_wrapper(i) for i in batches]

            direction_weight, direction_bias = 0, 0
            predictions = []

            for batch_gradient_weight, batch_gradient_bias, prediction in directions:
                direction_weight += batch_gradient_weight
                direction_bias += batch_gradient_bias
                predictions.append(prediction)

            direction_weight = (direction_weight * self.lr * inv_n) + (self.weight * (self.lr * inv_n * self.reg_para))
            direction_bias = direction_bias * self.lr * inv_n + (self.bias * (self.lr * inv_n * self.reg_para))

            self.weight -= direction_weight
            self.bias -= direction_bias

            nag_weight = self.weight + (self.weight - prev_weight) * self.mr
            nag_bias = self.bias + (self.bias - prev_bias) * self.mr

            if self.verbose > 0 and self.iter % self.verbose == 0:
                self.logger.info("Just finished iteration number %d " % (self.iter + 1))
                self.loss_list.append(self.loss(predictions, Y))
                self.true_loss_list.append(self.true_loss(X, Y))
                self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
            if self.save_weight > 0 and self.iter % self.save_weight == 0:
                self.weight_list.append(self.weight.copy())
                self.bias_list.append(self.bias.copy())

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
