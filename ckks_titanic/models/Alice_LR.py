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
            b_context: binary representation of the context_. context_.serialize()$
            b_X : list of binary representations of samples from CKKS vectors format
            b_Y : list of binary representations of labels from CKKS vectors format
            keys : keys of the samples which are passed to the subprocess. the local b_X[i] is the global X[keys[i]]. Useful to map predictions to true labels 
    This function is the first one to be passed in the input_queue queue of the process.
    It first deserialize the context_, passing it global,
    in the memory space allocated to the process
    Then the batch is also deserialize, using the context_,
    to generate a list of CKKS vector which stand for the encrypted samples on which the process will work
    """
    b_context = args[0]
    b_X = args[1]
    b_Y = args[2]
    global context_
    context_ = ts.context_from(b_context)
    global local_X_
    global local_Y_
    local_X_ = [ts.ckks_vector_from(context_, i) for i in b_X]
    local_Y_ = [ts.ckks_vector_from(context_, i) for i in b_Y]
    global local_keys
    local_keys = args[3]
    return 'Initialization done for process %s. Len of data : %i' % (
        multiprocessing.current_process().name, len(local_X_))


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

    bias = ts.ckks_vector_from(context_, b_bias)
    weight = ts.ckks_vector_from(context_, b_weight)

    predictions = forward(local_bias=bias, local_weight=weight, vec=local_X_)
    grads = backward(local_X_, predictions, local_Y_)
    b_grad_weight = grads[0].serialize()
    b_grad_bias = grads[1].serialize()
    b_predictions = [prediction.serialize() for prediction in predictions]
    return b_grad_weight, b_grad_bias, b_predictions, local_keys


class LogisticRegressionHE:
    """
        Model of logistic regression, performed on encrypted data, using homomorphic encryption, especially the CKKS scheme implemented in tenSEAL

    """

    def __init__(self,
                 init_weight,
                 init_bias,
                 context,
                 bob,
                 learning_rate=1,
                 max_epoch=100,
                 reg_para=0.5,
                 verbose=-1,
                 save_weight=-1,
                 n_jobs=1):
        """

            Constructor


            :param init_weight: CKKS vector. Initial weight
            :param init_bias: CKKS vector. Initial weight
            :param context: tenseal context_. Hold the public key, the relin key and the galois key. Those are mandatory to make computation and deserialization
            :param n_jobs: multiprocessing. Equal to the number of processes that will be created and launched
            :param bob : Actor, to who the weight will be sent to be refreshed.
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

        self.bob = bob
        if type(context) is bytes:
            self.context = ts.context_from(context)
            self.b_context = context
        else:
            self.context = context
            self.b_context = context.serialize()

        self.verbose = verbose
        self.save_weight = save_weight

        self.iter = 0
        self.num_iter = max_epoch
        self.reg_para = reg_para
        self.lr = learning_rate

        self.n_jobs = n_jobs

        if verbose > 0:
            self.loss_list = []
        if save_weight > 0:
            self.weight_list = []
            self.bias_list = []
        self.weight = init_weight
        self.bias = init_bias

    def set_bob(self, bob):
        """

        :param bob: Actor. Bob is needed for refreshing the weight, so as to fit the model.
                    If you do not need to fit the model (for instance if you load an already trained model) you can make predictions without BOB.
        """
        self.bob = bob

    def refresh(self, vector, return_bin=False):
        """
            The method refresh the depth of a ciphertext. It call the refresh function which aims to refresh ciphertext by preserving privacy
            :param return_bin: Boolean. If set to True, the function will return the serialized (binary) refreshed vector
            :param vector: CKKS vector, ciphertext
            :return: refreshed CKKS vector
        """
        try:
            self.bob.transmission(vector.serialize())
            if return_bin:
                return self.bob.reception()
            else:
                return ts.ckks_vector_from(self.context, self.bob.reception())
        except AttributeError:
            self.logger.critical('Bob is not provided, nobody is listening for refreshing the weight. '
                                 'You need to provide an Actor (see .setbob() method) '
                                 'and be sure that the actor is ready to refresh weight (see the 3-ap-Bob notebook')
            raise

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
        for y, prediction in zip(Y, enc_predictions):
            res -= y * self.__log(prediction) * inv_n
            res -= (1 - y) * self.__log(1 - prediction) * inv_n
        return res

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

    def fit(self, X, Y):
        """
        Train the model over encrypted data.

        :param X: list of CKKS vectors: encrypted samples (train set)
        :param Y: list of CKKS vectors: encrypted labels (train set)

        """

        keys = [[] for _ in range(self.n_jobs)]
        b_X = [[] for _ in range(self.n_jobs)]
        b_Y = [[] for _ in range(self.n_jobs)]
        Y_loss = Y

        if type(X[0]) is bytes:
            self.logger.info("Data already serialized")
            if self.verbose > 1:
                self.logger.info("Deserialization of labels for the future computations of the loss")
                Y_loss = [ts.ckks_vector_from(self.context, y) for y in Y]
            for i in range(len(X)):
                b_X[i % self.n_jobs].append(X[i])
                b_Y[i % self.n_jobs].append(Y[i])
                keys[i % self.n_jobs].append(i)
        else:
            self.logger.info("Starting serialization of data")
            ser_time = time.time()
            for i in range(len(X)):
                b_X[i % self.n_jobs].append(X[i].serialize())
                b_Y[i % self.n_jobs].append(Y[i].serialize())
                keys[i % self.n_jobs].append(i)
            self.logger.info("Data serialization done in %s seconds" % (time.time() - ser_time))
        inv_n = (1 / len(Y))
        self.logger.info("Initialization of %d workers" % self.n_jobs)
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
        self.logger.info("Initialization done in %s seconds" % (time.time() - init_work_timer))

        del b_X
        del b_Y

        while self.iter < self.num_iter:

            timer_iter = time.time()

            b_weight = self.refresh(self.weight, return_bin=True)
            b_bias = self.refresh(self.bias, return_bin=True)
            self.weight = ts.ckks_vector_from(self.context, b_weight)
            self.bias = ts.ckks_vector_from(self.context, b_bias)

            for q in list_queue_in:
                q.put((forward_backward, (b_weight, b_bias,)))

            direction_weight, direction_bias = 0, 0
            b_predictions = [0 for _ in range(len(X))]
            for _ in range(self.n_jobs):
                child_process_ans = queue_out.get()
                direction_weight += ts.ckks_vector_from(self.context, child_process_ans[0])
                direction_bias += ts.ckks_vector_from(self.context, child_process_ans[1])
                for prediction, key in zip(child_process_ans[2], child_process_ans[3]):
                    b_predictions[key] = prediction

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
                self.loss_list.append(
                    self.loss(Y_loss, (ts.ckks_vector_from(self.context, prediction) for prediction in b_predictions)))
                self.logger.info("Starting computation of the loss ...")
                self.logger.info('Loss : ' + str(self.loss_list[-1]) + ".")
            if self.save_weight > 0 and self.iter % self.save_weight == 0:
                self.weight_list.append(self.weight)
                self.bias_list.append(self.bias)

            self.iter += 1

        for q in list_queue_in:
            q.put('STOP')

        self.weight = self.refresh(self.weight)
        self.bias = self.refresh(self.bias)

        return self

    def predict(self, X):
        """
            Use the model to estimate a probability of a label .
            :param X: encrypted CKKS vector
            :return: encrypted prediction
        """
        return self.forward(X)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getstate__(self):

        attributes = {
            'b_context': self.b_context,
            'b_weight': self.weight.serialize(),
            'b_bias': self.bias.serialize(),
            'verbose': self.verbose,
            'save_weight': self.save_weight,
            'num_epoch': self.num_iter,
            'n_jobs': self.n_jobs,
            'reg_para': self.reg_para,
            'learning_rate': self.lr,
            'iter': self.iter
        }
        if self.verbose > 0:
            attributes['loss_list'] = [loss.Serialize() for loss in self.loss_list]
        if self.save_weight > 0:
            attributes['weight_list'] = [w.serialize() for w in self.weight_list]
            attributes['bias_list'] = [b.serialize() for b in self.bias_list]

        return attributes

    def __setstate__(self, attributes):

        self.b_context = attributes['b_context']
        self.context = ts.context_from(self.b_context)
        self.weight = ts.ckks_vector_from(self.context, attributes['b_weight'])
        self.bias = ts.ckks_vector_from(self.context, attributes['b_bias'])
        self.verbose = attributes['verbose']
        self.save_weight = attributes['save_weight']
        self.num_iter = attributes['num_epoch']
        self.n_jobs = attributes['n_jobs']
        self.reg_para = attributes['reg_para']
        self.lr = attributes['learning_rate']
        self.iter = attributes['iter']
        if self.verbose > 0:
            self.loss_list = [ts.ckks_vector_from(self.context, loss) for loss in attributes['loss_list']]
        if self.save_weight > 0:
            self.weight_list = [ts.ckks_vector_from(self.context, w) for w in attributes['weight_list']]
            self.bias_list = [ts.ckks_vector_from(self.context, b) for b in attributes['bias_list']]
