import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2 as cv

def main():
    
    
    
    inputs=layers.Input(shape=(160, 60, 3))
    
    '''
    # shared convolutional layer
    c1 = layers.Conv2D(20, (5,5),   strides=(2,2),padding='same',activation='relu', use_bias=False)(inputs)
    mp1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(mp1)
    avg2 = layers.AveragePooling2D()(c2)
    c3 = layers.Conv2D(50, (5,5), strides=(2,2),padding='same', activation='relu',use_bias=False)(avg2)
    avg3=layers.AveragePooling2D()(c3)
    '''
    

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
    
    
    model = Model(inputs=inputs, outputs=[ a_fc2, b_fc2, c_fc2, d_fc2 ] )
    
      
    print "architecture created"
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    