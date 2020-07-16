import numpy as np
# TODO : mettre à jour le code proprement avec la convention equancy
# TODO : mettre à jour le code avec vers la version crytpé, d'organisation plus propre

def config_loss_entropy():
    def activation(x):
        return 1 / (1 + np.exp(-x))

    def oracle(X, Y_hat, theta, acti=activation):
        Y = acti(np.dot(X, theta)).reshape(X.shape[0], 1)
        critere = - np.mean((Y_hat == 0) * np.log(Y) + (Y_hat == 1).T * np.log(1 - Y))

        grad = np.dot(X.T, (Y - Y_hat)) / Y.shape[0]
        return critere, grad

    return activation, oracle


def config_loss_entropy_MMsig3():
    def activation(x):  # approximation of sigmoid, with minmax approx, degree 3
        return -0.004 * np.power(x, 3) + 0.197 * x + 0.5

    def oracle(X, Y_hat, theta, acti=activation):
        Y = acti(np.dot(X, theta)).reshape(X.shape[0], 1)
        critere = - np.mean((Y_hat == 0) * np.log(Y) + (Y_hat == 1).T * np.log(1 - Y))
        grad = np.dot(X.T, (Y - Y_hat)) / Y.shape[0]
        return critere, grad

    return activation, oracle






class LogisticRegression:
    def __init__(self, config=config_loss_entropy, lr=0.001, num_iter=100000, crit_norm_grad=0.005 ,fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.activation, self.oracle = config()
        self.verbose = verbose
        self.crit_norm_grad = crit_norm_grad
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, Y, theta=None):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # init_weight init
        if theta is None:
            self.theta = np.zeros((X.shape[1], 1))
        else:
            self.theta = theta
        critere, gradient = self.oracle(X, Y, self.theta)
        self.CRITERE = [critere]
        self.NGRADIENT = [np.linalg.norm(gradient)]
        self.iter = 0
        while np.linalg.norm(self.NGRADIENT[-1]) > self.crit_norm_grad and self.iter < self.num_iter:
            critere, gradient = self.oracle(X, Y, self.theta)
            self.theta -= gradient * self.lr
            self.CRITERE.append(critere)
            self.NGRADIENT.append(np.linalg.norm(gradient))
            self.iter += 1
            if (self.verbose and self.iter % 1000 == 0):
                print("loss :" + str(self.CRITERE[-1]))

    def predict(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.activation(np.dot(X, self.theta))

    def accuracy(self, X, Y):
        return 1-np.mean(np.power(Y - self.predict(X), 2))
