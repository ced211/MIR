import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

#See https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b

def get_train_test(split_ratio=0.7, random_state=42):
    file = open("C:\\Users\\cedri\\Documents\\nsynth-train\\labels","r")
    labels = json.load(file)
    mfccs = "C:\\Users\\cedri\\Documents\\nsynth-train\\mfccs\\"
    i = 0
    for filename, label in labels.items():
        filename,_,_ = filename.partition(".wav")
        if i == 0:
            X = np.load(mfccs + filename + ".npy")
            print(X.shape)
            y = label
            i = 1
            continue
        x = np.load(mfccs + filename + '.npy')
        X = np.vstack((X, x))
        y = np.append(y,label)
        if i == 1587:
            check = x
        i += 1
    X = X.reshape((len(y),-1,x.shape[1]))
    print(X.shape)
    print(y.shape)
    assert X.shape[0] == len(y)
    if(False in np.equal(X[1587],check)):
        print("error")
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def plotX(X):
    fig, ax = plt.subplots()
    min = np.min(X)
    max = np.max(X)
    ax.matshow(X, cmap=plt.cm.Blues)

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_train_test()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    print("Construct model")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(y_train_hot.shape[1], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))
