import numpy as np
import BPNeuralNetwork
import mnistloader


def trainMNIST():
    loader = mnistloader.mnistloader(
        'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    (data, label) = loader.getImages(5000, 0)
    bp = BPNeuralNetwork.BPNetwork((784, 15, 10))
    print "Begin to train..."
    bp.miniBatch(data, label, 5000)
    bp.save('paras.dat')
    print "\ntrain done, predict 100 images"
    (testdata, testlabel) = loader.getImages(1000, 5000)
    res = bp.predict(testdata)
    for i in range(res.shape[1]):
        tmp = np.zeros(res.shape[0])
        tmp[np.argmax(res[:, i])] = 1
        res[:, i] = tmp
    print np.sum(np.abs(res - testlabel))
