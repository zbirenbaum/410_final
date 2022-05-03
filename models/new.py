from tensorflow import keras as ks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential, Model

def build(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(8,kernel_size=(3,3),strides=(1,1), input_shape=input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16,kernel_size=(3,3),strides = (1,1)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32,kernel_size=(3,3),strides = (1,1)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,kernel_size=(3,3),strides=(1,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(256,kernel_size=(2,2),strides=(1,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.3))
    model.add(Dense(classes, activation='softmax', name='fc' + str(classes)))
    model.compile(optimizer="Adadelta",loss="categorical_crossentropy",metrics=["accuracy"])
    return model
