import numpy as np
import pandas as pd

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsize):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsize)))
    return w.astype(np.float32)

def relu(x):
    # f(x) = max(0, x)
    return x * (x > 0)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis = 1, keepdims = True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    #uses targets to index Y instead of multiplying by a matrix of many 0s
    N = len(T)
    return -np.log(Y[np.arrange(N), T]).mean()

def error_rate(targets, predictions):
    return np.mean(target != predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind

def getRawData(balance_ones = True, limit_count = 40000):
    #Image Size: 48x48
    expressions = []
    pixels = []
    first = True
    total_images = 0
    for row in open('fer2013.csv'):
        if first:
            first = False
        else:
            row_values = row.split(',')
            expressions.append(int(row[0]))
            pixels.append([int(p) for p in row_values[1].split(" ")])
            total_images += 1
        if total_images > limit_count:
            break
    pixels = np.array(pixels) / 255.0
    expressions = np.array(expressions)

    if balance_ones:
        #clas imbalance problem
        X1, Y1 = pixels[expressions != 1, :], expressions[expressions != 1]
        X2 = pixels[expressions == 1, :]
        X2 = np.repeat(X2, 9, axis = 0)
        pixels = np.vstack([X1, X2])
        expressions = np.concatenate((Y1, [1]*len(X1)))

    return pixels, expressions

def getImage(limit_to = 40000):
    X, Y = getRawData(limit_count = limit_to)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])

    return np.array(X) / 255.0, np.array(Y)


def labelCodes():
    labels =  {
                0: 'Angry',
                1: 'Disgust',
                2: 'Fear',
                3: 'Happy',
                4: 'Sad',
                5: 'Surprise',
                6: 'Neutral',
                }
    return labels

def crossValidation(model, X, Y, K=5):
    #split data into K parts
    X, Y = np.shuffle(X, Y)
    sz = len(Y) / K
    errors = []
    for k in xrange(K):
        x_train = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        y_train = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        x_test = X[k*sz:(k*sz + sz), :]
        y_test = Y[k*sz:(k*sz + sz)]

        model.fit(x_train, y_train)
        err = model.score(x_test, y_test)
        errors.append(err)
    #print "errors:", errors
    return np.mean(errors)
