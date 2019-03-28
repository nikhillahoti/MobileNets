import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import keras
from keras import Model
import time
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D

def createFiles():
    fTemp = open("fTemp.txt", "w")

    """
    counter = [1, 100, 1000]
    for i in range(3):
        iterr = 1 + i
        for j in range(224):
            for k in range(224):
                fTemp.write(str(iterr) + "\n")
                iterr += 1
    """

    img_width = 224
    img_height = 224
    fileName = "Dog.jpg"
    img = image.load_img(fileName, target_size=(img_width, img_height))
    X = image.img_to_array(img)

    for i in range(3):
        for j in range(224):
            for k in range(224):
                fTemp.write(str(X[j][k][i]) + "\n")
    fTemp.close()

    fTemp = open("fWeights.txt", "w")
    for i in range(32):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    fTemp.write(str(i * 10 + j) + "\n")
    fTemp.close()

def checkCalculations():

    img_width = 224
    img_height = 224
    fileName = "Dog.jpg"
    img = image.load_img(fileName, target_size=(img_width, img_height))
    X = image.img_to_array(img)


    """
    X[:,:,0] = 1
    X[:,:,1] = 100
    X[:,:,2] = 1000

    counter = [1, 100, 1000]
    for i in range(3):
        for j in range(224):
            for k in range(224):
                X[j][k][i] = counter[i]

    """

    """
    for i in range(3):
        iterr = 1 + i
        for j in range(224):
            for k in range(224):
                X[j][k][i] = iterr
                iterr += 1
    """

    X = np.expand_dims(X, axis=0)
    print(X.shape)

    mobile = keras.applications.mobilenet.MobileNet(weights="imagenet")

    model = Model(mobile.input, mobile.layers[2].output)
    # print(model.layers[2].get_weights())
    W = model.layers[2].get_weights()[0]
    #W[:,:,:,:] = 1

    """
    for i in range(32):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    W[j][k][l][i] = (i * 10 + j)
    print(W[0][0][1][0])
    print(W[0][0][1][1])
    print(W[0][0][1][20])
    """

    W = np.expand_dims(W, axis=0)
    print(W.shape)
    model.layers[2].set_weights(W)
    output = model.predict(X)
    print(output.shape)

    # Save Output to file
    fOutput = open("First_Layer_Output.txt", "w")
    for i in range(len(output)):
        for j in range(len(output[0][0][0])):
            for k in range(len(output[0])):
                for l in range(len(output[0][0])):
                    fOutput.write(str(output[i][k][l][j]) + "\n")
    fOutput.close()
    print("Writing in file Complete!!!")

def getDict():
    dict = {}
    fIn = open('output.txt', "r")
    content = fIn.readlines()
    for i in range(len(content)):
        if not content[i] in dict:
            dict[content[i]] = True
    fIn.close()
    #print(dict)

import time
start = time.time()
createFiles()
checkCalculations()
end = time.time()
print("Total Time -->", (end - start))

print("hahah")
#getDict()

