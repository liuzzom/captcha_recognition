import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import cv2 as cv
import os
from numpy import array
 
def getLabels(dirPath):
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    imgNames.sort()
 
    # list containing first letters
    firstLetters = [int(name[0]) for name in imgNames]
    secondLetters = [int(name[1]) for name in imgNames]
 
    # conversion in numpy array
    firstLetters = array(firstLetters)
    secondLetters = array(secondLetters)
 
    # conversion in categorical numpy array
    firstLetters = tf.keras.utils.to_categorical(firstLetters)
    secondLetters = tf.keras.utils.to_categorical(secondLetters)
 
    return [firstLetters,secondLetters]
 
def getImages(dirPath):
    # TO DO: rename to be more evocative
 
    # get all image names
    imgNames = [name for name in os.listdir(dirPath) if name.endswith(".png")]
    imgNames.sort()
    images = [image_to_scalegray(dirPath, imgName) for imgName in imgNames]
    images = array(images)
    return images
 
def image_to_scalegray(path_dir,img_name):
 
    # transform a rgb image to its scalegray representation
 
    path = path_dir + img_name
    img = cv.imread(path, 0)
    _,img=cv.threshold(img,210,255,cv.THRESH_BINARY)  
    img = img.reshape([60, 160, 1])
    return img
 
def main():
 
    inputs=layers.Input(shape=(60, 160, 1))
 
 
    print("creating layers...")
 
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
    b_c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=True)(b_avg2)
    b_avg3=layers.AveragePooling2D(strides=(2,2))(b_c3)
    
    #Concat & flatten layers
 
    
    #Chain A
    a_x=layers.concatenate([a_avg3,b_avg3])
    
    b_x=layers.concatenate([a_avg3,b_avg3])
    
    
    #Concat & flatten layers
    a_fl = layers.Flatten()(a_x)
    b_fl = layers.Flatten()(b_x)
    
   
    
    # Fully connnected layers
 
    #Chain A
    a_fc1 = layers.Dense(512, activation='relu', use_bias=True)(a_fl)
    a_fc2 = layers.Dense(10,activation='softmax', use_bias=True)(a_fc1)
    
    #Chain B
    b_fc1 = layers.Dense(512, activation='relu', use_bias=True)(b_fl)
    b_fc2 = layers.Dense(10, activation='softmax', use_bias=True)(b_fc1)
    
    
 
    print("layers created\ncreating model...")
    model = Model(inputs=inputs, outputs=(a_fc2,b_fc2))
    print("model created")
 
    print("compiling...")
    sgd = optimizers.SGD()
    model.compile(sgd, loss='mean_squared_error', metrics=['accuracy'])
    print("compiled")
    
    # images and labels for training and validation
    print("loading images and labels for training...")
    trainLabels =  getLabels("./train_2/")
    trainImages = getImages('./train_2/')
    print("done")
 
    print(type(trainLabels))
    print(type(trainImages))
 
    # training and validation
    print("training...")
    model.fit(x=trainImages, y=trainLabels, batch_size=256, epochs=5, validation_split=0.1)
    print("done")
 
    # images and labels for testing
    testLabels = getLabels("./test_2/")
    testImages = getImages("./test_2/")
 
    # test
    score=model.evaluate(x=testImages, y=testLabels, batch_size=256)
 
    print(score)
    
 
if __name__ == "__main__":
    main()