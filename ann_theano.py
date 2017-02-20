import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from utilities import getRawData, getBinaryData, error_rate, init_weight_and_bias, relu
from sklearn.utilites import shuffle

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W%s' %self.id)
        self.b = theano.shared(W, 'b%s' %self.id)

    def forward(self, X):
        return relu(X.dot(self.W) + self.b)
