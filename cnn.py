from data_import import import_data
import numpy as np
from tensorflow import keras as ks
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
# import librosa.display

def get_encoding(labels):
    return {
        "encode": {label: i for i, label in enumerate(list(set(labels)))},
        "decode": {i: label for i, label in enumerate(list(set(labels)))}
    }
def encode_labels(labels, encoding_map):
    return [encoding_map["encode"][label] for label in labels]
def decode_label(encoded_label, encoding_map):
    return encoding_map["decode"][encoded_label]

def get_X_y (path):
    X, labels = import_data(path)
    encoding_map = get_encoding(labels)
    y = np.array(encode_labels(labels, encoding_map))
    return X, y, encoding_map

X, y, encoding_map = get_X_y('./data/images_original')
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


model = ks.Sequential()
model.add(ks.layers.Conv2D(filters=4, kernel_size=(128, 128), activation='relu', input_shape=(256,256,3)))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Conv2D(filters=8, kernel_size=(64, 64), activation='relu', input_shape=(256,256,3)))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Conv2D(filters=16, kernel_size=(32, 32), activation='relu', input_shape=(256,256,3)))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Conv2D(filters=32, kernel_size=(8, 8), activation='relu', input_shape=(256,256,3)))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(units=120, activation='relu'))
model.add(ks.layers.Dense(units=84, activation='relu'))
model.add(ks.layers.Dense(units=10, activation = 'softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(np.array(X), np.array(y), batch_size=100, epochs=10, validation_split=0.15)
