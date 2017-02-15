import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct


def getInt(nbytes):
    n = 0
    for i in nbytes:
        n = n * 256 + struct.unpack('B', i)[0]
    return n


def foo(loader):
    image = np.ndarray((0, 0))
    n = -1
    while(image.shape[0] < 100):
        n += 1
        if(loader.getLabel(n) != 0):
            continue
        (im, la) = loader.getImage(n)
        if(image.shape[0] == 0):
            image = im.reshape((1, 28 * 28))
        else:
            image = np.vstack((image, im.reshape((1, 28 * 28))))
    return image


class mnistloader:
    def __init__(self, imagename, labelname):
        self.imagef = open(imagename)
        self.labelf = open(labelname)
        self.imagef.seek(4)
        imgs = self.imagef.read(4)
        self.nimgs = getInt(imgs)
        rows = self.imagef.read(4)
        cols = self.imagef.read(4)
        self.shape = (getInt(rows), getInt(cols))

    def close(self):
        if(not self.imagef.closed):
            self.imagef.close()
        if(not self.labelf.closed):
            self.labelf.close()

    def __del__(self):
        self.close()

    def getImage(self, ind):
        self.imagef.seek(16 + ind * self.shape[0] * self.shape[1])
        self.labelf.seek(8 + ind)
        buf = self.imagef.read(self.shape[0] * self.shape[1])
        image = [struct.unpack('B', i)[0] for i in buf]
        image = np.asarray(image)
        image = image.reshape(self.shape[0], self.shape[1])
        buf = self.labelf.read(1)
        label = struct.unpack('B', buf)[0]
        return image, label

    def getLabel(self, ind):
        self.labelf.seek(8 + ind)
        buf = self.labelf.read(1)
        return struct.unpack('B', buf)[0]

    def getImages(self, n, start):
        data = np.zeros((self.shape[0] * self.shape[1], n))
        label = np.zeros((10, n))
        for i in range(n):
            (img, l) = self.getImage(i + start)
            data[:, i] = img.reshape(784) / 255.0
            label[l, i] = 1
        return data, label

    def plotImage(self, ind):
        (image, label) = self.getImage(ind)
        plt.imshow(image, mpl.cm.gray)
        plt.show()
