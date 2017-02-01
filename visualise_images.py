import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utilities import getRawData, getImage, labelCodes

def visualise():
    images, expressions =  getImage(limit_to = 300)
    #300 images contain all kinds of expressions
    labels = labelCodes()

    done = [0, 0, 0, 0, 0, 0, 0]
    count = 7
    os.mkdir("Visualisations")
    for image, expression in zip(images, expressions):
        image = image[0]
        if done[expression] == 0:
            done[expression] = 1
            plt.imshow(image, cmap = 'gray')
            image_name = "Visualisations/" + labels[expression] + ".png"
            plt.savefig(image_name)
            #plt.show()
            count -= 1
        if count == 0:
            break


if __name__ == '__main__':
    visualise()
