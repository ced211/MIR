import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm

import time
import keras
from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from Mfcc_freq_seq import FqMfccSeq
import copy
from HistoryMem import HistoryMem

if __name__ == "__main__":

    history = HistoryMem(filepath = "..\\models\\conv-spectrum-fq\\history-reg ")
    train = FqMfccSeq("..\\spectrum-train\\","..\\nsynth-train")
    valid = FqMfccSeq("..\\spectrum-valid\\","..\\nsynth-valid")
    model = Sequential()
    model.add(Conv1D(32, kernel_size= 4, activation='relu', input_shape=(train.x_shape[0],1)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\conv-spectrum-fq-reg\\models-{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train, epochs=50, verbose=1, validation_data=valid,callbacks=[checkpoint,history])
