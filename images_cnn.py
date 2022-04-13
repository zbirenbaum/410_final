import librosa
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from shallow_evaluate import model_Evaluate
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow.keras.layers as Layers
import librosa.display
from PIL import Image


data_path = './data/images_original'
labels = os.listdir(data_path)
for 
data_dict={ genre:[] for genre in genres}
data=[]
y = []
for genre in genres:
    for file in os.listdir(PATH+genre+'/'):
        if file!='jazz00054.png':
            image = Image.open(PATH+genre+'/'+file)
            data_dict[genre].append(image)
            data.append(np.array(image))
    y.append(genre)


print(np.array(data).shape)

y = [genres.index(gen) for gen in y]
print(np.array(y).shape)
model = ks.Sequential()
model.add(ks.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(288,432,4)))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(ks.layers.AveragePooling2D())
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(units=120, activation='relu'))
model.add(ks.layers.Dense(units=84, activation='relu'))
model.add(ks.layers.Dense(units=10, activation = 'softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(np.array(data), np.array(y), batch_size=100, epochs=10, validation_split=0.15)

