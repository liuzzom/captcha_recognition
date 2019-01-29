import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

import cv2 as cv
from sklearn.preprocessing import MultiLabelBinarizer
import os
from numpy import array
from cv2.cv2 import THRESH_BINARY

def charToInt(char):
    # TO DO: change bad name
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return charset.find(char)

def getLabels(dirPath):
    
        
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    
    # lists containing letters
    firstLetters = [name[0] for name in imgNames]
    secondLetters = [name[1] for name in imgNames]
    thirdLetters = [name[2] for name in imgNames]
    fourthLetters = [name[3] for name in imgNames]

    
    # labelling
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    classes = array([char for char in charset])
    
    mlb = MultiLabelBinarizer(classes=classes)
    
    firstLetters = mlb.fit_transform(y=firstLetters)
    secondLetters = mlb.fit_transform(y=secondLetters)
    thirdLetters = mlb.fit_transform(y=thirdLetters)
    fourthLetters = mlb.fit_transform(y=fourthLetters)
    
    
    return [firstLetters, secondLetters, thirdLetters, fourthLetters]

def getImages(dirPath):
    # TO DO: rename to be more evocative
    
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    images = array([image_to_scalegray(dirPath, imgName) for imgName in imgNames])
    return images

def image_to_scalegray(path_dir,img_name):
    
    # transform a rgb image to its scalegray representation
    path = path_dir + img_name
    img = cv.imread(path, 0)
    cv.threshold(img,127,255,cv.THRESH_BINARY)
    img = img.reshape([60, 160, 1])
    return img
    
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
    
    #print "creating layers..."

    # 4 separate convolutional layers
    
    # Chain A
    a_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same', activation='relu', use_bias=True)(inputs)
    a_mp1 = layers.MaxPooling2D()(a_c1)
    a_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(a_mp1)
    a_avg2 = layers.AveragePooling2D()(a_c2)
    a_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same',  activation='relu',use_bias=True)(a_avg2)
    a_avg3=layers.AveragePooling2D()(a_c3)
    
    # Chain B
    b_c1 = layers.Conv2D(20, (5,5), strides=(2,2), padding='same',activation='relu', use_bias=True)(inputs)
    b_mp1 = layers.MaxPooling2D(strides=(2,2))(b_c1)
    b_c2 = layers.Conv2D(32, (5,5), strides=(2,2), padding='same',activation='relu',use_bias=True)(b_mp1)
    b_avg2 = layers.AveragePooling2D(strides=(2,2))(b_c2)
    b_c3 = layers.Conv2D(50, (5,5), strides=(2,
                                             2),padding='same', activation='relu',use_bias=True)(b_avg2)
    b_avg3=layers.AveragePooling2D(strides=(2,2))(b_c3)
    
    # Chain C
    c_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same',activation='relu', use_bias=True)(inputs)
    c_mp1 = layers.MaxPooling2D(strides=(2,2))(c_c1)
    c_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(c_mp1)
    c_avg2 = layers.AveragePooling2D(strides=(2,2))(c_c2)
    c_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(c_avg2)
    c_avg3=layers.AveragePooling2D(strides=(2,2))(c_c3)
    
    # Chain D
    d_c1 = layers.Conv2D(20, (5,5), strides=(2,2),padding='same', activation='relu', use_bias=True)(inputs)
    d_mp1 = layers.MaxPooling2D(strides=(2,2))(d_c1)
    d_c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(d_mp1)
    d_avg2 = layers.AveragePooling2D(strides=(2,2))(d_c2)
    d_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(d_avg2)
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
    a_fc1 = layers.Dense(512, activation='relu', use_bias=True)(a_fl)
    a_fc2 = layers.Dense(62,activation='softmax', use_bias=True)(a_fc1)
    
    #Chain B
    b_fc1 = layers.Dense(512, activation='relu', use_bias=True)(b_fl)
    b_fc2 = layers.Dense(62, activation='softmax', use_bias=True)(b_fc1)
    
    #Chain C
    c_fc1 = layers.Dense(512, activation='relu', use_bias=True)(c_fl)
    c_fc2 = layers.Dense(62, activation='softmax', use_bias=True)(c_fc1)
    
    #Chain D
    d_fc1 = layers.Dense(512, activation='relu', use_bias=True)(d_fl)
    d_fc2 = layers.Dense(62, activation='softmax', use_bias=True)(d_fc1)
    
    #print "layers created\ncreating model..."
    model = Model(inputs=inputs, outputs=[ a_fc2, b_fc2, c_fc2, d_fc2 ] )
    #print "model created"
    
    #print "compiling..."
    sgd = optimizers.SGD()
    model.compile(sgd, loss='mean_squared_error', metrics=['accuracy'])
    #print "compiled"

    # images and labels for training and validation
    #print "loading images and labels for training..."
    trainLabels =  getLabels("./train/")
    trainImages = getImages("./train/")
    #print "done"
    
    # training and validation
    #print "training..."
    model.fit(x=trainImages, y=trainLabels, batch_size=1000, epochs=2, validation_split=0.1)
    #print "done"
    
    # images and labels for testing
    testLabels = getLabels("./test/")
    testImages = getImages("./test/")
    
    # test
    score=model.evaluate(x=testImages, y=testLabels, batch_size=1000)
    
    print(score)
    
if __name__ == "__main__":
    main()