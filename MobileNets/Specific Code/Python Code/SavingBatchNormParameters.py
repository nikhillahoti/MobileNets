import numpy as np
import keras
from keras.models import Model
import math as m

mobile = keras.applications.mobilenet.MobileNet(weights="imagenet")

model = Model(mobile.input, mobile.layers[4].output)

batchNorm = mobile.layers[3].get_weights()


print(batchNorm)
print("------")

fMean = open("data/LayerOne/First_Layer_Mean.txt", "w")
fSD = open("data/LayerOne/First_Layer_StanDev.txt", "w")
fGamma = open("data/LayerOne/First_Layer_Gamma.txt", "w")
fBeta = open("data/LayerOne/First_Layer_Beta.txt", "w")


for i in range(len(batchNorm[0])):
    fGamma.write(str(float(batchNorm[0][i])) + "\n")

for i in range(len(batchNorm[1])):
    fBeta.write(str(float(batchNorm[1][i])) + "\n")

for i in range(len(batchNorm[2])):
    fMean.write(str(float(batchNorm[2][i])) + "\n")

for i in range(len(batchNorm[3])):
    fSD.write(str(m.sqrt(float(batchNorm[3][i]) + 0.001)) + "\n")


print(batchNorm[0])
print(batchNorm[1])
print(batchNorm[2])
print(batchNorm[3])

fMean.close()
fSD.close()
fGamma.close()
fBeta.close()
