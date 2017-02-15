#BPNeuralNetwork
A python framwork of BP neural network.<br />
We add a bias neuron at the begining of each layer instead of using bias term for each neuron.<br />
the activate function is sigmod function, which is `1.0/(1.0+exp(-z))`.<br />
the cost function is `-sigma(y*log(a)+(1-y)*log(1-a))+regularization_term`, where y is the actual value of training data 
and a is the output of network while sigma is aggregation operation.<br />
Here we use MNIST dataset to test our network. so I wrote `mnistloader.py` to handle MNIST file.<br />

##Usage
* Construction

```python
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
```
we accept the shape as input parameter. we predefined some parameters.<br />


* Train

we offer two ways of train data, mini-batch and onlie training.
```python
def miniBatch(self, data, label, data_size, batch_size=100)
def onlineTraining(self, data, label, data_size)
```

* Save and load

we use shelve to save and load parameters<br />
```python
def save(self, filename)
def load(self, filename)
```