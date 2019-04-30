import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from HistoryMem import HistoryMem
import time
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from mfcc_seq import Mfcc
import copy
import datetime

#See https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b

if __name__ == "__main__":

    history = HistoryMem(filepath = "..\\models\\conv-spectrum\\history ")

    now = datetime.datetime.now()
    train = Mfcc("..\\spectrum-train\\","..\\nsynth-train")
    valid = Mfcc("..\\spectrum-valid\\","..\\nsynth-valid")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(train.x_shape[0], train.x_shape[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\conv-spectrum\\models-{epoch:02d}" + now.strftime("%d %H:%M") + ".hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    history = model.fit_generator(train, epochs=40, verbose=1, validation_data=valid,callbacks=[checkpoint,history])
    model.save("complete_model.model")