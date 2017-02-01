import numpy as np
import pandas as pd

def getRawData(balance_ones = True):
    #Image Size: 48x48
    expressions = []
    pixels = []
    first = True
    for row in open('Data/fer2013.csv'):
        if first:
            first = False
        else:
            row_values = line.split(',')
            expressions.append(int(row[0]))
            pixels.append([int(p) for p in row[1].split()])

        pixels = np.array(pixels) / 255.0
        expressions = np.array(expressions)
        if balance_ones:
            pass #class balancing later
        return pixels, expressions

def getImage():
    X, Y = getRawData()
    N, D = X.Shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y
