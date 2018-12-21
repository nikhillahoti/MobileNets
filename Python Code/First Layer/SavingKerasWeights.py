
import numpy as np
import keras
from keras.models import Model
import cv2

mobile = keras.applications.mobilenet.MobileNet(weights="imagenet")
model = Model(mobile.input, mobile.output)



def saveWeights():
    print("Started saving -> ")
    f = open("First_Layer_Weights.txt", "w")
    layerr = model.layers[2].get_weights()

    print(len(layerr[0][0][0][0]))
    print(len(layerr))
    for y in range(len(layerr)):
        for l in range(len(layerr[0][0][0][0])):
            for k in range(len(layerr[0][0][0])):
                for i in range(len(layerr[0])):
                    for j in range(len(layerr[0][0])):
                        f.write(str(layerr[y][i][j][k][l]) + "\n")
    f.close()

    print("----")
    print(len(layerr))
    print(layerr[0][0][0][0][0])

    import numpy as np
    arr = np.array(layerr)
    print(arr.shape)

    print("Number of Layers: ", len(model.layers))
    print("Saved successfully!!")

def saveBiases():
    f = open("FirstLayerBiases.txt", "w")
    layerr = model.layers[2].params[1]

    print(model.layers[3])
    print("Total Layers --> ", len(model.layers))
    print(" ---> ", len(layerr))
    print(" ---> ", len(layerr[0]))
    print(np.array(layerr).shape)

def getInputImage():
    from keras.preprocessing import image
    img = image.load_img("Dog.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def getOutput():
    from keras.applications import imagenet_utils
    mobile = keras.applications.mobilenet.MobileNet(weights="imagenet")
    preprocessed_image = getInputImage()
    #print(preprocessed_image)
    print(preprocessed_image.shape)

    """
    fileInput = open("MBInput.txt", "w")
    for i in range(len(preprocessed_image)):
        for j in range(len(preprocessed_image[0][0][0])):
            for k in range(len(preprocessed_image[0][0])):
                for l in range(len(preprocessed_image[0])):
                    fileInput.write(str(preprocessed_image[i][l][k][j]) + "\n")
    fileInput.close()

    #predictions = mobile.predict(preprocessed_image)
    #results = imagenet_utils.decode_predictions(predictions)
    #print(results)

    model = Model(mobile.input, mobile.layers[2].output)
    output =  model.predict(preprocessed_image)
    print(output.shape)
    file = open("MyOutput.txt", "w")
    for i in range(len(output)):
        for j in range(len(output[0][0][0])):
            for k in range(len(output[0][0])):
                for l in range(len(output[0])):
                    file.write(str(output[i][k][l][j]) + "\n")
    file.close()
    """
#getOutput()
saveWeights()
#saveBiases()
