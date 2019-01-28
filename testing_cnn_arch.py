import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers  
import cv2 as cv
import os
from numpy import array

def charToInt(char):
    str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return str.find(char)

def getLabels(dirPath):
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    
    # lists containing letters
    firstLetters = [name[0] for name in imgNames]
    secondLetters = [name[1] for name in imgNames]
    thirdLetters = [name[2] for name in imgNames]
    fourthLetters = [name[3] for name in imgNames]

    # conversion into lists containing numbers
    firstLetters = [charToInt(char) for char in firstLetters]
    secondLetters = [charToInt(char) for char in secondLetters]
    thirdLetters = [charToInt(char) for char in thirdLetters]
    fourthLetters = [charToInt(char) for char in fourthLetters]

    # conversion in numpy arrays
    firstLetters = array(firstLetters)
    secondLetters = array(secondLetters)
    thirdLetters = array(thirdLetters)
    fourthLetters = array(fourthLetters)
    
    # conversion in categorical numpy arrays
    firstLetters = tf.keras.utils.to_categorical(firstLetters, num_classes=62)
    secondLetters = tf.keras.utils.to_categorical(secondLetters, num_classes=62)
    thirdLetters = tf.keras.utils.to_categorical(thirdLetters, num_classes=62)
    fourthLetters = tf.keras.utils.to_categorical(fourthLetters, num_classes=62)
    
    
    return (firstLetters, secondLetters, thirdLetters, fourthLetters)

def getImages(dirPath):
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    
    images = []
    
    for imgName in imgNames:
        path = dirPath + imgNames[0]
        img = cv.imread(path, 0)
        images.append(img)
    
    images = array(images)
    return images
    
def main():
    
    inputs=layers.Input(shape=(60, 160, 1))
    
    '''
    # shared convolutional layer
    c1 = layers.Conv2D(20, (5,5),   strides=(2,2),padding='same',activation='relu', use_bias=False)(inputs)
    mp1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(mp1)
    avg2 = layers.AveragePooling2D()(c2)
    c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(avg2)
    avg3=layers.AveragePooling2D()(c3)
    '''
    
    print "creating layers..."

    # 4 separate convolutional layers
    
    # Chain A
    a_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same', activation='relu', use_bias=False)(inputs)
    a_mp1 = layers.MaxPooling2D()(a_c1)
    a_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(a_mp1)
    a_avg2 = layers.AveragePooling2D()(a_c2)
    a_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same',  activation='relu',use_bias=False)(a_avg2)
    a_avg3=layers.AveragePooling2D()(a_c3)
    
    # Chain B
    b_c1 = layers.Conv2D(20, (5,5), strides=(2,2), padding='same',activation='relu', use_bias=False)(inputs)
    b_mp1 = layers.MaxPooling2D(strides=(2,2))(b_c1)
    b_c2 = layers.Conv2D(32, (5,5), strides=(2,2), padding='same',activation='relu',use_bias=False)(b_mp1)
    b_avg2 = layers.AveragePooling2D(strides=(2,2))(b_c2)
    b_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(b_avg2)
    b_avg3=layers.AveragePooling2D(strides=(2,2))(b_c3)
    
    # Chain C
    c_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same',activation='relu', use_bias=False)(inputs)
    c_mp1 = layers.MaxPooling2D(strides=(2,2))(c_c1)
    c_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(c_mp1)
    c_avg2 = layers.AveragePooling2D(strides=(2,2))(c_c2)
    c_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(c_avg2)
    c_avg3=layers.AveragePooling2D(strides=(2,2))(c_c3)
    
    # Chain D
    d_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same', activation='relu', use_bias=False)(inputs)
    d_mp1 = layers.MaxPooling2D(strides=(2,2))(d_c1)
    d_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(d_mp1)
    d_avg2 = layers.AveragePooling2D(strides=(2,2))(d_c2)
    d_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(d_avg2)
    d_avg3=layers.AveragePooling2D(strides=(2,2))(d_c3)
    
    # Concat & flatten layers
    
    #Chain A
    a_x=layers.concatenate([a_avg3,b_avg3])
    
    #Chain B
    b_x=layers.concatenate([a_avg3,b_avg3,c_avg3])
    
    #Chain C
    c_x=layers.concatenate([b_avg3,c_avg3,d_avg3])
    
    #Chain D
    d_x=layers.concatenate([c_avg3,d_avg3])
    
    a_fl = layers.Flatten()(a_x)
    b_fl = layers.Flatten()(b_x)
    c_fl = layers.Flatten()(c_x)
    d_fl = layers.Flatten()(d_x)
    
    
    # Fully connnected layers
    
    
    #Chain A
    a_fc1 = layers.Dense(512, activation='relu', use_bias=None)(a_fl)
    a_fc2 = layers.Dense(62,activation='softmax', use_bias=None)(a_fc1)
    
    #Chain B
    b_fc1 = layers.Dense(512, activation='relu', use_bias=None)(b_fl)
    b_fc2 = layers.Dense(62, activation='softmax', use_bias=None)(b_fc1)
    
    #Chain C
    c_fc1 = layers.Dense(512, activation='relu', use_bias=None)(c_fl)
    c_fc2 = layers.Dense(62, activation='softmax', use_bias=None)(c_fc1)
    
    #Chain D
    d_fc1 = layers.Dense(512, activation='relu', use_bias=None)(d_fl)
    d_fc2 = layers.Dense(62, activation='softmax', use_bias=None)(d_fc1)
    
    print "layers created\ncreating model..."
    model = Model(inputs=inputs, outputs=[ a_fc2, b_fc2, c_fc2, d_fc2 ] )
    print "model created"
    
    print "compiling..."
    sgd = optimizers.SGD()
    model.compile(sgd, loss='mean_squared_error', metrics=['accuracy'])
    print "compiled"

    # images and labels for training and validation
    print "loading images and labels for training..."
    trainLabels = []
    trainFirstLetters, trainSecondLetters, trainThirdLetters, trainFourthLetters =  getLabels("./train/")
    trainLabels.append(trainFirstLetters)
    trainLabels.append(trainSecondLetters)
    trainLabels.append(trainThirdLetters)
    trainLabels.append(trainFourthLetters)
    trainImages = getImages("./train/")
    print "done"
    
    # training and validation
    print "training..."
    model.fit(x=trainImages, y=trainLabels, batch_size=1000, epochs=2, validation_split=0.1)
    print "done"
    
    # images and labels for testing
    testLabels = []
    testFirstLetters, testSecondLetters, testThirdLetters, testFourthLetters =  getLabels("./test/")
    testLabels.append(testFirstLetters)
    testLabels.append(testSecondLetters)
    testLabels.append(testThirdLetters)
    testLabels.append(testFourthLetters)
    testImages = getImages("./test/")
    
    # test
    model.evaluate(x=testImages, y=testLabels, batch_size=1000)
    
if __name__ == "__main__":
    main()