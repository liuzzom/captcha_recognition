from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,Input,concatenate

def main():
    
    inputs=Input(shape=(160,60,3))
    
    # shared convolutional layer
    
    c1 = Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False)(inputs)
    mp1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False)(mp1)
    avg2 = AveragePooling2D()(c2)
    c3 = Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False)(avg2)
    avg3=AveragePooling2D()(c3)
    
    # 4 separate convolutional layers
    
    # Chain A
    a_c1 = Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False)(avg3)
    a_mp1 = MaxPooling2D()(a_c1)
    a_c2 = Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False)(a_mp1)
    a_avg2 = AveragePooling2D()(a_c2)
    a_c3 = Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False)(a_avg2)
    a_avg3=AveragePooling2D()(a_c3)
    
    # Chain B
    b_c1 = Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False)(avg3)
    b_mp1 = MaxPooling2D()(b_c1)
    b_c2 = Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False)(b_mp1)
    b_avg2 = AveragePooling2D()(b_c2)
    b_c3 = Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False)(b_avg2)
    b_avg3=AveragePooling2D()(b_c3)
    
    # Chain C
    c_c1 = Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False)(avg3)
    c_mp1 = MaxPooling2D()(c_c1)
    c_c2 = Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False)(c_mp1)
    c_avg2 = AveragePooling2D()(c_c2)
    c_c3 = Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False)(c_avg2)
    c_avg3=AveragePooling2D()(c_c3)
    
    # Chain D
    d_c1 = Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False)(avg3)
    d_mp1 = MaxPooling2D()(d_c1)
    d_c2 = Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False)(d_mp1)
    d_avg2 = AveragePooling2D()(d_c2)
    d_c3 = Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False)(d_avg2)
    d_avg3=AveragePooling2D()(d_c3)
    
    # Concat & flatten layers
    
    #Chain A
    a_x=concatenate([a_avg3,b_avg3])
    
    b_x=concatenate([a_avg3,b_avg3,c_avg3])
    
    c_x=concatenate([b_avg3,c_avg3,d_avg3])
    
    d_x=concatenate([c_avg3,d_avg3])
    
    
    
    
    
    
    '''
    model=Sequential()
    
    #1st convolutional shared layer
    #model.add(Conv2D(20, (5,5), (2,2), activation='relu', use_bias=False,input_shape=(160,60,3)))
    
    
    #max pooling of 1st conv layer 
    model.add(MaxPooling2D())
    
    #2nd convolutinal shared layer
    model.add(Conv2D(32, (5,5), (2,2), activation='relu',use_bias=False))
    
    #avg pooling of 2nd conv layer
    model.add(AveragePooling2D())
              
    #3rd convolutinal shared layer
    model.add(Conv2D(50, (5,5), (2,2), activation='relu',use_bias=False))
    
    #avg pooling of 3rd conv layer
    model.add(AveragePooling2D())
    
    model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, target_tensors)
    '''
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    