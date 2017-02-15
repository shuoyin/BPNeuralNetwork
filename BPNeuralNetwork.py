import numpy as np
import shelve


def sigmod(z):
    val = 1.0 / (1.0 + np.exp(-z))
    deri = val * (1.0 - val)
    return deri, val


def costFun(thetalist, a, y, minbatch_size, reguP):
    """the cost function is:
    -sigma(y*log(a)+(1-y)*log(1-a)) + regularization_term
    where sigma means aggregation operation
    return the value of fun"""
    value = y * np.log(a) + (1 - y) * np.log(1 - a)
    value = -np.sum(value) / minbatch_size
    regu = 0
    for theta in thetalist:
        regu += np.sum(np.power(theta, 2))
    value = value + reguP * regu / (2 * minbatch_size)
    return value


class BPNetwork:
    def __init__(self, shape=0):
        self.shape = shape
        self.reguP = 0.2  # regularization parameter lambda
        self.stepL = 3  # step length or called learning rate
        self.maxiter = 1000
        self.eplison = 0.00001
        self.paras = []
        if shape == 0:
            return
        for i in range(len(shape) - 1):
            theta = np.random.randn(shape[i + 1], shape[i] + 1)
            self.paras.append(theta)
        self.zlist = [None] * (len(shape) - 1)
        self.alist = [None] * len(shape)

    def __predict__(self, data, actif, num):
        self.alist[0] = data
        app = np.asarray([1] * num)
        if num > 1:
            app = app.reshape(1, num)
        for i in range(len(self.paras)):
            self.alist[i] = np.append(app, self.alist[i], axis=0)
            self.zlist[i] = np.dot(self.paras[i], self.alist[i])
            self.alist[i + 1] = actif(self.zlist[i])[1]

    def predict(self, data):
        if data.ndim == 1:
            num = 1
        else:
            num = data.shape[1]
        self.__predict__(data, sigmod, num)
        return self.alist[-1]

    def trainBatch(self, data, label, size):
        length = len(self.zlist)  # length of zlist, also deltalist
        deltalist = [None] * length
        self.__predict__(data, sigmod, size)  # feed forward
        # back propagation
        deltalist[length - 1] = self.alist[length] - label
        for i in range(length - 2, -1, -1):
            D = self.alist[i + 1] * (1 - self.alist[i + 1]) * \
                np.dot(self.paras[i + 1].T, deltalist[i + 1])
            deltalist[i] = D[1:, :]
        for i in range(length):
            diag = np.eye(self.paras[i].shape[1])
            diag[0, 0] = 0
            deri = (np.dot(deltalist[i], self.alist[i].T) +
                    self.reguP * np.dot(self.paras[i], diag)) / size
            self.paras[i] = self.paras[i] - self.stepL * deri
        return costFun(self.paras, self.alist[-1], label, size, self.reguP)

    def train(self, data, label, data_size, batch_size):
        new_cost = -1
        old_cost = 0
        iteration = 0
        # go through all training data
        while abs(new_cost - old_cost) > self.eplison and iteration < self.maxiter:
            iteration += 1
            print "\niter: ", iteration
            # train every batch
            for start in range(0, data_size, batch_size):
                old_cost = new_cost
                end = min(start + batch_size, data_size)
                new_cost = self.trainBatch(data[:, start:end], label[
                                           :, start:end], end - start)
                print "cost function: ", new_cost,
                if abs(new_cost - old_cost) < self.eplison:
                    break

    def miniBatch(self, data, label, data_size, batch_size=100):
        self.train(data, label, data_size, batch_size)

    def onlineTraining(self, data, label, data_size):
        self.train(data, label, data_size, 1)

    def save(self, filename):
        if filename[-4:] != '.dat':
            filename = filename + '.dat'
        f = shelve.open(filename)
        f['paras'] = self.paras
        f.close()

    def load(self, filename):
        f = shelve.open(filename)
        self.paras = f['paras']
        f.close()
        shapel = [self.paras[0].shape[1] - 1]
        for theta in self.paras:
            shapel.append(theta.shape[0])
        self.shape = tuple(shapel)
        self.zlist = [None] * (len(self.shape) - 1)
        self.alist = [None] * len(self.shape)
