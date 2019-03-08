import numpy as np

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

#mean squared error loss
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) **2).mean()

class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self,inputs):
        #Weight inputs add bias then sigmoid activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

