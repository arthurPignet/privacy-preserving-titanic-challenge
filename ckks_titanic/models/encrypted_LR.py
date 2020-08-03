import logging
import multiprocessing
import time
import tenseal as ts


# multiprocessing utils functions

def worker(input_queue, output_queue):
    """
    This functions turns on the process until a string 'STOP' is found in the input_queue queue

    It takes every couple (function, arguments of the functions) from the input_queue queue, and put the result into the output queue
    """
    for func, args in iter(input_queue.get, 'STOP'):
        result = func(*args)
        output_queue.put(result)


# noinspection PyGlobalUndefined
def initialization(*args):
    """
    :param:tuple : (b_context, b_X, b_X,keys)
            b_context: binary representation of the context. context.serialize()$
            b_X : list of binary representations of samples from CKKS vectors format
            b_Y : list of binary representations of labels from CKKS vectors format
            keys : keys of the samples which are passed to the subprocess. the local b_X[i] is the global X[keys[i]]. Useful to map predictions to true labels 
    This function is the first one to be passed in the input_queue queue of the process.
    It first deserialaze the context, passing it global,
    in the memory space allocated to the process
    Then the batch is also deserialize, using the context,
    to generate a list of CKKS vector which stand for the encrypted samples on which the proces will work
    """
    b_context = args[0]
    b_X = args[1]
    b_Y = args[2]
    global context
    context = ts.context_from(b_context)
    global local_X
    global local_Y
    local_X = [ts.ckks_vector_from(context, i) for i in b_X]
    local_Y = [ts.ckks_vector_from(context, i) for i in b_Y]
    global local_keys
    local_keys = args[3]
    return 'Initialization done for process %s. Len of data : %i' % (
        multiprocessing.current_process().name, len(local_X))


def forward_backward(*args):
    b_weight = args[0]
    b_bias = args[1]

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

    def forward(local_weight, local_bias, vec, mult_coeff=1):
        """
            Compute forward propagation on a CKKS vector (or a list of CKKS vectors)
            :param local_bias: encrypted bias used in forward propagation
            :param local_weight: encrypted weight used in forward propagation
            :param vec: CKKS vector or list of CKKS vector on which we want to make local_predictions (ie forward propagation
            :param mult_coeff: The return is equivalent to forward(x) * mult_coeff, but save one homomorph multiplication
            :return: encrypted prediction or list of encrypted local_predictions
        """

        if type(vec) == list:
            temp = [i.dot(local_weight) + local_bias for i in vec]
            return [sigmoid(i, mult_coeff=mult_coeff) for i in temp]
        else:
            res = vec.dot(local_weight) + local_bias
            return sigmoid(res, mult_coeff=mult_coeff)

    def backward(X, local_predictions, Y):
        """
            Compute the backpropagation on a given encrypted batch
            :param X: list of encrypted (CKKS vectors). Features of the data on which the gradient will be computed (backpropagation)
            :param local_predictions: list of encrypted CKKS vectors. Label predictions (forward propagation) on the data on which the gradient will be computed (backpropagation)
            :param Y: list of encrypted CKKS vectors. Label of the data on which the gradient will be computed (backpropagation)
            :return: Tuple of 2 CKKS vectors. Encrypted direction of descent for weight and bias"
        """
        if type(X) == list:
            err = local_predictions[0] - Y[0]
            grad_weight = X[0] * err
            grad_bias = err
            for i in range(1, len(X)):
                err = local_predictions[i] - Y[i]
                grad_weight += X[i] * err
                grad_bias += err
            return grad_weight, grad_bias
        else:
            err = local_predictions - Y
            grad_weight = X * err
            grad_bias = err

            return grad_weight, grad_bias

    bias = ts.ckks_vector_from(context, b_bias)
    weight = ts.ckks_vector_from(context, b_weight)

    predictions = forward(local_bias=bias, local_weight=weight, vec=local_X)
    grads = backward(local_X, predictions, local_Y)
    b_grad_weight = grads[0].serialize()
    b_grad_bias = grads[1].serialize()
    b_predictions = [pred.serialize() for pred in predictions]
    return (b_grad_weight, b_grad_bias, b_predictions, local_keys)


class LogisticRegressionHE:
    """
        Model of logistic regression, performed on encrypted data, using homomorphic encryption, especially the CKKS scheme implemented in tenSEAL

    """

    def __init__(self,
                 init_weight,
                 init_bias,
                 refresh_function,
                 context,
                 confidential_kwarg,
                 accuracy=None,
                 learning_rate=1,
                 momentum_rate=0,
                 max_epoch=100,
                 reg_para=0.5,
                 verbose=-1,
                 save_weight=-1,
                 n_jobs=1):
        """

            Constructor


            :param init_weight: CKKS vector. Initial weight
            :param init_bias: CKKS vector. Initial weight
            :param refresh_function: function. Refresh ciphertext
            :param confidential_kwarg: dict. Will be passed as **kwarg to refresh, loss and accuracy functions. Contain confidential data which are needed by those functions.
            :param accuracy: function. Compute accuracy
            :param learning_rate: float. learning rate
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
        self.context = context

        self.verbose = verbose
        self.save_weight = save_weight

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = learning_rate
        self.mr = momentum_rate

        self.n_jobs = n_jobs
        self.b_context = context.serialize()

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

    def loss(self, Y, enc_predictions):
        """

            This function is here to homomorphically estimate the cross entropy loss

            1-NOTE : this function could be parallelize, as we do not need the result for the next epoch.

            :parameters
            ------------

            self : model
            Y : encrypted labels of the dataset on which the loss will be computed
            enc_predictions : iterator of encrypted model predictions on the dataset on which the loss will be computed
                              One can denote that the forward propagation (predictions) has to be done before.
            :returns
            ------------

            loss : float (rounded to 3 digits)


        """
        self.weight = self.refresh(self.weight)
        self.bias = self.refresh(self.bias)
        inv_n = 1 / len(Y)
        res = (self.reg_para * inv_n / 2) * (self.weight.dot(self.weight) + self.bias * self.bias)
        for y, pred in zip(Y,enc_predictions):
            res -= y * self.__log(pred) * inv_n
            res -= (1 - y) * self.__log(1 - pred) * inv_n
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

    @staticmethod
    def backward(X, predictions, Y):
        """
            Compute the backpropagation on a given encrypted batch
            :param X: list of encrypted (CKKS vectors). Features of the data on which the gradient will be computed (backpropagation)
            :param predictions: list of encrypted CKKS vectors. Label predictions (forward propagation) on the data on which the gradient will be computed (backpropagation)
            :param Y: list of encrypted CKKS vectors. Label of the data on which the gradient will be computed (backpropagation)
            :return: Tuple of 2 CKKS vectors. Encrypted direction of descent for weight and bias"
        """
        if type(X) == list:
            err = predictions[0] - Y[0]
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

    def forward_backward_wrapper(self, arg):
        """
        Wrapper for forward_backward, which expands the parameter tuple to forward_backward
        :param arg: Tuple, (X,Y) with X standing for a list of encrypted (CKKS vectors). Features of the data on which predictions will be made, (forward propagation) and then the gradient will be computed (backpropagation)
                                  and Y standing for a list of encrypted CKKS vectors. Label of the data on which the gradient will be computed (backpropagation)
        :return:
                Tuple with 3 CKKS vectors. Encrypted batch_gradient for weight and bias, and batch predictions.
        """
        return self.forward_backward(*arg)

    def forward_backward(self, X, Y):
        """
        Perform forward propagation, and then backward propagation.

        :param X: list of encrypted (CKKS vectors). Features of the data on which predictions will be made, (forward propagation) and then the gradient will be computed (backpropagation)
        :param Y: list of encrypted CKKS vectors. Label of the data on which the gradient will be computed (backpropagation)
        :return: : Tuple with 3 CKKS vectors. Encrypted batch_gradient for weight and bias, and batch predictions.

        """
        predictions = self.forward(X)
        grads = LogisticRegressionHE.backward(X, predictions, Y)
        return grads[0], grads[1], predictions

    def fit(self, X, Y):
        """
        Train the model over encrypted data.

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)

        """
        self.logger.info("Starting serialization of data")
        ser_time = time.time()
        keys = [[] for _ in range(self.n_jobs)]
        b_X = [[] for _ in range(self.n_jobs)]
        b_Y = [[] for _ in range(self.n_jobs)]
        for i in range(len(X)):
            b_X[i % self.n_jobs].append(X[i].serialize())
            b_Y[i % self.n_jobs].append(Y[i].serialize())
            keys[i % self.n_jobs].append(i)
        self.logger.info("Data serialization done in %s seconds" %(time.time()-ser_time))
        inv_n = (1 / len(Y))
        self.logger.info("Initialization of %d workers" %self.n_jobs)
        init_work_timer = time.time()
        list_queue_in = []
        queue_out = multiprocessing.Queue()
        init_tasks = [(initialization, (self.b_context, x, y, key)) for x, y, key in zip(b_X, b_Y, keys)]
        for init in init_tasks:
            list_queue_in.append(multiprocessing.Queue())
            list_queue_in[-1].put(init)
            multiprocessing.Process(target=worker, args=(list_queue_in[-1], queue_out)).start()
        log_out = []
        for _ in range(self.n_jobs):
            log_out.append(queue_out.get())
            logging.info(log_out[-1])
        self.logger.info("Initialization done in %s seconds" %(time.time()-init_work_timer))

        while self.iter < self.num_iter:
            timer_iter = time.time()
            self.weight = self.refresh(self.weight)
            self.bias = self.refresh(self.bias)

            b_weight = self.weight.serialize()
            b_bias = self.bias.serialize()
            for q in list_queue_in:
                q.put((forward_backward, (b_weight, b_bias,)))
            direction_weight, direction_bias = 0, 0
            b_predictions = [0 for _ in range(len(X))]
            for _ in range(self.n_jobs):
                log_out.append(queue_out.get())
                direction_weight += ts.ckks_vector_from(self.context, log_out[-1][0])
                direction_bias += ts.ckks_vector_from(self.context, log_out[-1][1])
                for pred, key in zip(log_out[-1][2], log_out[-1][3]):
                    b_predictions[key]=pred
                             
            direction_weight = (direction_weight * self.lr * inv_n) + (self.weight * (self.lr * inv_n * self.reg_para))
            direction_bias = direction_bias * self.lr * inv_n + (self.bias * (self.lr * inv_n * self.reg_para))

            self.weight -= direction_weight
            self.bias -= direction_bias

            self.logger.info(
                "Just finished iteration number %d in  %s seconds. " % (
                    self.iter, time.time() - timer_iter))

            if self.verbose > 0 and self.iter % self.verbose == 0:
                self.weight = self.refresh(self.weight)
                self.bias = self.refresh(self.bias)
                self.loss_list.append(self.loss(Y,(ts.ckks_vector_from(self.context, pred) for pred in b_predictions)))
                self.logger.info("Starting computation of the loss ...")
                self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
            if self.save_weight > 0 and self.iter % self.save_weight == 0:
                self.weight_list.append(self.weight)
                self.bias_list.append(self.bias)

            self.iter += 1
        return self

    def predict(self, X):
        """
            Use the model to predict a label.
            :param X: encrypted CKKS vector
            :return: encrypted prediction
        """
        return self.forward(X)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
