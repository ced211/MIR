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
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\labels","r")
    labels = json.load(file)
    mfccs = "C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\MIR-master\\mfccs\\"
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
	
def get_train_test_valid(split_ratio=0.7, random_state=42):
    # train set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\instruments","r")
    instruments = json.load(file)
    mfccs = "C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\mfccs-train\\"
    i = 0
    for filename, instrument in instruments.items():
        filename,_,_ = filename.partition(".wav")
        if i == 0:
            X_train = np.load(mfccs + filename + ".npy")
            print(X_train.shape)
            y_train = instrument
            i = 1
            continue
        x = np.load(mfccs + filename + '.npy')
        X_train = np.vstack((X_train, x))
        y_train = np.append(y_train,instrument)
        if i == 1587:
            check = x
        i += 1
    X_train = X_train.reshape((len(y_train),-1,x.shape[1]))
    print(X_train.shape)
    print(y_train.shape)
    assert X_train.shape[0] == len(y_train)
    if(False in np.equal(X_train[1587],check)):
        print("error")
		
	# validation set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-valid\\instruments","r")
    instruments = json.load(file)
    mfccs = "C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\mfccs-valid\\"
    i = 0
    for filename, instrument in instruments.items():
        filename,_,_ = filename.partition(".wav")
        if i == 0:
            X_valid = np.load(mfccs + filename + ".npy")
            print(X_valid.shape)
            y_valid = instrument
            i = 1
            continue
        x = np.load(mfccs + filename + '.npy')
        X_valid = np.vstack((X_valid, x))
        y_valid = np.append(y_valid,instrument)
        if i == 1587:
            check = x
        i += 1
    X_valid = X_valid.reshape((len(y_valid),-1,x.shape[1]))
    print(X_valid.shape)
    print(y_valid.shape)
    assert X_valid.shape[0] == len(y_valid)
    if(False in np.equal(X_valid[1587],check)):
        print("error")
	
	# test set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-test\\instruments","r")
    instruments = json.load(file)
    mfccs = "C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\mfccs-test\\"
    i = 0
    for filename, instrument in instruments.items():
        filename,_,_ = filename.partition(".wav")
        if i == 0:
            X_test = np.load(mfccs + filename + ".npy")
            print(X_test.shape)
            y_test = instrument
            i = 1
            continue
        x = np.load(mfccs + filename + '.npy')
        X_test = np.vstack((X_test, x))
        y_test = np.append(y_test,instrument)
        if i == 1587:
            check = x
        i += 1
    X_test = X_test.reshape((len(y_test),-1,x.shape[1]))
    print(X_test.shape)
    print(y_test.shape)
    assert X_test.shape[0] == len(y_test)
    if(False in np.equal(X_test[1587],check)):
        print("error")	
		
    return X_train, X_test, X_valid, y_train, y_test, y_valid

def plotX(X):
    fig, ax = plt.subplots()
    min = np.min(X)
    max = np.max(X)
    ax.matshow(X, cmap=plt.cm.Blues)

if __name__ == "__main__":

    X_train, X_test, X_valid, y_train, y_test, y_valid = get_train_test_valid()

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
    model.save("complete_model.model")