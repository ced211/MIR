import numpy as np
from keras.layers import Input, Dense, Add, Lambda,Dropout,Flatten
from keras.models import Model
from mfcc_seq_time_freq import Seq_time_freq
from keras.layers import Dense, Dropout
from HistoryMem import HistoryMem
import keras

if __name__ == "__main__":

    history = HistoryMem(filepath = "..\\models\\time_freq-spectrum\\history ")
    train = Seq_time_freq("..\\spectrum-train\\","..\\nsynth-train",1000,False,0.7)
    valid = Seq_time_freq("..\\spectrum-train\\","..\\nsynth-train",1000,True,0.7)

    input_train = Input(shape=[train.x_shape[0]])
    input_valid = Input(shape=[valid.x_shape[0]])

    x = Dense(128,activation = 'relu')(input_train)
    x = Dropout(0.25)(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dropout(0.25)(x)
    
    out = Dense(11,activation='softmax')(x)

    model = Model(inputs=input_train, outputs = out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\time_freq-spectrum\\models-{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train, epochs=50, verbose=1, validation_data=valid,callbacks=[checkpoint,history],shuffle=True)
