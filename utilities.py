import numpy as np
import pandas as pd

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
        pass #class balancing later

    return pixels, expressions

def getImage(limit_to = 40000):
    X, Y = getRawData(limit_count = limit_to)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

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
