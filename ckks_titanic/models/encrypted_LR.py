import logging


class LogisticRegressionHE:
    """
        Model of logistic regression, performed on encrypted data, using homomorphic encryption, especially the CKKS scheme implemented in tenSEAL

    """

    def __init__(self,
                 init_weight,
                 init_bias,
                 refresh_function,
                 confidential_kwarg,
                 accuracy=None,
                 lr=1,
                 max_epoch=100,
                 reg_para=0.5,
                 verbose=-1,
                 save_weight=-1,
                 ):
        """

            Constructor


            :param init_weight: CKKS vector. Initial weight
            :param init_bias: CKKS vector. Initial weight
            :param refresh_function: function. Refresh ciphertext
            :param confidential_kwarg: dict. Will be passed as **kwarg to refresh, loss and accuracy functions. Contain confidential data which are needed by those functions.
            :param accuracy: function. Compute accuracy
            :param lr: float. learning rate
            :param max_epoch: int. number of epoch to be performed
            :param reg_para: float. regularization parameter
            :param verbose: int. number of epoch were the loss is not computed, nor printed.
                            Every <verbose> epoch, the loss (and error) will be logged
                            If set to -1, the loss will not be computed nor stored at all

            :param save_weight: int. number of epoch were the weight will be stored.
                                Every <save_weight> epoch, the weight will be logged in weight_list
                                If set to -1, the weight will not be saved

        """
        self.logger = logging.getLogger(__name__)

        self.refresh_function = refresh_function
        self.confidential_kwarg = confidential_kwarg
        self.accuracy_function = accuracy

        self.verbose = verbose
        self.save_weight = save_weight

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = lr

        if verbose > 0:
            self.loss_list = []
        if save_weight > 0:
            self.weight_list = []
            self.bias_list = []
        self.weight = init_weight
        self.bias = init_bias

    def refresh(self, vector):
        """
            The method refresh the depth of a ciphertext. It call the refresh function which aims to refresh ciphertext by preserving privacy
            :param vector: CKKS vector, ciphertext
            :return: refreshed CKKS vector
        """
        return self.refresh_function(vector, **self.confidential_kwarg)

    def loss(self, X, Y):
        enc_prediction = self.forward(X)
        res = (self.reg_para / 2) * (self.weight.dot(self.weight) + self.bias * self.bias)
        for i in range(len(enc_prediction)):
            res += Y[i] * self.__log(enc_prediction[i])
            res += (1 - Y[i]) * self.__log(1 - enc_prediction[i])
        return res

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
    def __log(enc_x, mult_coeff=1):
        poly_coeff = [-3.69404813, 13.30907268, -19.06853265, 9.63445963]
        return enc_x.polyval([i * mult_coeff for i in poly_coeff])

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

    def fit(self, X, Y):
        """
        Train the model over encrypted data.

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)

        """
        while self.iter < self.num_iter:

            self.weight = self.refresh(self.weight)
            self.bias = self.refresh(self.bias)
            # refreshing the init_weight and the init_bias to avoid scale out of bound
            # encrypted gradient descent
            encrypted_prediction = self.forward(X)  # we can add batching later
            direction_weight, direction_bias = self.backward(X, encrypted_prediction, Y)
            self.bias -= direction_bias
            self.weight -= direction_weight

            if self.verbose > 0 and self.iter % self.verbose == 0:
                self.logger.info("iteration number %d is starting" % (self.iter + 1))
                self.loss_list.append(self.loss())
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
