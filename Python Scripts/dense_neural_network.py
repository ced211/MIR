import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm

import time
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from mfcc_seq import Mfcc
import copy

if __name__ == "__main__":

    train = Mfcc("..\\mfccs-train\\","..\\nsynth-train")
    valid = Mfcc("..\\mfccs-valid\\","..\\nsynth-valid")
    model = Sequential()
    model.add(Dense(100, activation='relu',input_shape = (20,50,1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(11, activation='softmax'))
    print("compiling")
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint("..\\dense-models-{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train, epochs=train.__len__(), verbose=1, validation_data=valid,callbacks=[checkpoint])
    model.save("complete_model.model")