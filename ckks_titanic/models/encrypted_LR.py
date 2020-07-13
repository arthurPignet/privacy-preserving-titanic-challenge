import time

import numpy as np
import tenseal as ts

from sklearn.model_selection import train_test_split





def crytp_array(X, local_context):
    res = []
    for vec in X:
        res.append(ts.ckks_vector(local_context, vec))
    return res


class LogisticRegressionHE:
    def __init__(self, local_context=None, lr=0.01, num_iter=100000,
                 verbose=False, reg_para=0.5, safety=False):

        if not safety:
            self.DIR_NE = []
            self.CRITERE_NE = []
            print("Warning :  The data will be decrypted during the process, the protocol is not safe")
            assert local_context is not None, "The protocol chosen (safety is set to false) need the context to refresh and decrypt ciphertext"
        self.safety = safety
        self.num_iter = num_iter
        self.verbose = verbose
        self.CRITERE = []
        self.NGRADIENT = []
        self.weight = None
        self.weight_ne = None
        self.bias = None
        self.bias_ne = None
        self.iter = 0
        self.context = local_context
        self.reg_para = reg_para
        self.lr = lr

    def refresh(self, vector):
        return ts.ckks_vector(self.context, vector.decrypt())

    @staticmethod
    def sigmoid(enc_x, mult_coef=1):
        return (enc_x * enc_x) * (enc_x * (-0.004 * mult_coef)) + enc_x * (0.197 * mult_coef) + (0.5 * mult_coef)

    def forward(self, vec, mult_coef=1):
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid(i, mult_coef=mult_coef) for i in temp]

        else:

            res = vec.dot(self.weight) + self.bias

            return LogisticRegressionHE.sigmoid(res, mult_coef=mult_coef)

    def forward_ne(self, vec, mult_coef=1):
        if type(vec) == list:
            temp = [i.dot(self.weight) + self.bias for i in vec]
            return [LogisticRegressionHE.sigmoid(i, mult_coef=mult_coef) for i in temp]

        else:
            res = vec.dot(self.weight_ne)
            return LogisticRegressionHE.sigmoid(res, mult_coef=mult_coef)

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
        # initialization to 0.
        self.weight_ne = [0. for _ in range(X[0].size())]
        self.weight = ts.ckks_vector(self.context, self.weight_ne)
        self.weight_ne = np.array(self.weight_ne)
        self.bias_ne = [0.]
        self.bias = ts.ckks_vector(self.context, self.bias_ne)
        self.bias_ne = np.array(self.bias_ne)
        # initial values
        for i in range(len(X)):
            print(X[i].decrypt())
        print(X_ne)
        for i in range(len(Y)):
            print(Y[i].decrypt())
        print(Y_ne)
        encrypted_prediction = self.forward(X)  # we can add batching later
        prediction = self.forward_ne(X_ne)
        direction_weight, direction_bias = self.backward(X, encrypted_prediction, Y)
        ne_direction_weight, ne_direction_bias = self.backward_ne(X_ne, prediction, Y_ne)
        ne_loss = np.log(prediction).dot(Y_ne)
        ne_loss += (1 - np.array(Y_ne)).T.dot(np.log(1 - np.array(prediction)))
        ne_loss += (self.reg_para / 2) * self.weight_ne.dot(self.weight_ne)
        self.CRITERE_NE.append(ne_loss)
        self.DIR_NE.append((ne_direction_weight, ne_direction_bias))
        self.NGRADIENT.append((direction_weight.decrypt(), direction_bias.decrypt()))
        print("encrypted grad")
        print(direction_weight.decrypt())
        print('Unencrypted grad')
        print(ne_direction_weight)
        print(
            "error %d" % (np.sum(np.power((np.array(direction_weight.decrypt()) - ne_direction_weight), 2)) / np.sum(
                np.power(ne_direction_weight, 2))))

        while self.iter < self.num_iter:
            if not self.safety:
                self.weight = self.refresh(self.weight)
                self.bias = self.refresh(self.bias)
                # refreshing the weight and the bias to avoid scale out of bound
                # TODO FIX UNSECURED PROTOCOL
            encrypted_prediction = self.forward(X)  # we can add batching later
            prediction = self.forward_ne(X_ne)
            direction_weight, direction_bias = self.backward(X, encrypted_prediction, Y)
            ne_direction_weight, ne_direction_bias = self.backward_ne(X_ne, prediction, Y_ne)
            self.weight -= direction_weight
            self.weight_ne -= ne_direction_weight
            self.bias -= direction_bias
            self.bias_ne -= ne_direction_bias
            self.iter += 1
            if self.verbose and self.iter % 1 == 0:
                print("iteration number %d is starting" % self.iter)
                if not self.safety:
                    ne_loss = np.log(prediction).dot(Y_ne)
                    ne_loss += (1 - np.array(Y_ne)).T.dot(np.log(1 - np.array(prediction)))
                    ne_loss += (self.reg_para / 2) * self.weight_ne.dot(self.weight_ne)
                    self.CRITERE_NE.append(ne_loss)
                    self.DIR_NE.append((ne_direction_weight, ne_direction_bias))
                    self.NGRADIENT.append((direction_weight.decrypt(), direction_bias.decrypt()))
                    print("encrypted grad")
                    print(direction_weight.decrypt())
                    print('Unencrypted grad')
                    print(ne_direction_weight)
                    print(
                        "error %d" % (np.sum(
                            np.power((np.array(direction_weight.decrypt()) - ne_direction_weight), 2)) / np.sum(
                            np.power(ne_direction_weight, 2))))

    def predict(self, X):
        return self.forward(X)

    def plain_accuracy(self, X_test, Y_test):
        prediction = self.forward(X_test)
        err = Y_test[0] - prediction[0]
        for i in range(1, len(X_test)):
            err += np.float(np.abs(Y_test[i] - prediction[i]) < 0.5)
        return np.mean(err)

    def encrypted_accuracy(self, X_test, Y_test):
        prediction = self.forward(X_test)
        err = Y_test[0] - prediction[0]
        for i in range(1, len(X_test)):
            err += np.float(np.abs((Y_test[i].decrypt() - prediction[i]).decrypt()) < 0.5)
        return err / len(X_test)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

#TODO mettre tout ca dans un notebook à part
if __name__ == "__main__":
    print("Uploading and preprocessing the data")
    # paths
    # data_path = "./data/" #whole data set
    data_path = "./data_quick_demo/"  # subset of the data set, with 15 train_samples and 5 test_samples
    # loading and prepossessing of the data
    raw_train, raw_test = preprocessing.data_import(data_path)
    train, submission_test = preprocessing.preprocessing(raw_train, raw_test)
    # splitting in train (85%) and test(15%) sets
    train, test = train_test_split(train, test_size=0.15)
    train_labels = train.Survived
    test_labels = test.Survived
    train_features = train.drop("Survived", axis=1)
    test_features = test.drop("Survived", axis=1)
    # adding bias

    print("Definition of safety parameters")
    timer = time.time()
    # context = ts.context(ts.SCHEME_TYPE.CKKS, 32768,
    # coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60])

    context = ts.context(ts.SCHEME_TYPE.CKKS, 16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60])
    # context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40])
    context.global_scale = pow(2, 40)
    print("Done. " + str(time.time() - timer) + " seconds")
    print('Generation of Galois Key...')
    timer = time.time()
    context.generate_galois_keys()
    print("Done. " + str(time.time() - timer) + " seconds")
    print("Data encryption...")
    timer = time.time()
    encrypted_X = crytp_array(train_features.to_numpy(), context)
    encrypted_Y = crytp_array(train_labels.to_numpy().reshape((-1, 1)), context)
    encrypted_test_X = crytp_array(test_features.to_numpy(), context)
    encrypted_test_Y = crytp_array(test_labels.to_numpy().reshape((-1, 1)), context)
    print("Done. " + str(time.time() - timer) + " seconds")
    print("Training starting")
    model = LogisticRegressionHE(local_context=context, safety=False, verbose=True, num_iter=40)
    model.fit(encrypted_X, encrypted_Y, X_ne=train_features.to_numpy(), Y_ne=train_labels.to_numpy().reshape((-1, 1)))
    print("Training done. " + str(time.time() - timer) + " seconds")
    model.weight = ts.ckks_vector(model.context, model.weight.decrypt())
    for x, y in zip(encrypted_test_X, encrypted_test_Y):
        print("label : " + str(y.decrypt()[0]))
        print("predict : " + str(model.predict(x).decrypt()))
    acc = model.encrypted_accuracy(encrypted_test_X, encrypted_test_Y)
    print("Accuracy : %s " % acc[0])
