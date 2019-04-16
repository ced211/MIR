from keras.applications.vgg16 import VGG16
import numpy as np
from mfcc_seq import Mfcc
from keras.layers import Input, Dense, Concatenate, Lambda,Dropout,Flatten
from keras.models import Model
import keras

train = Mfcc("..\\spectrum-train\\","..\\nsynth-train",32)
valid = Mfcc("..\\spectrum-valid\\","..\\nsynth-valid",32)

input = Input(shape=(train.x_shape[0],train.x_shape[1],1))
img_conc = Concatenate()([input,input, input]) 
vgg = VGG16(weights=None, include_top=False,
        input_shape =(train.x_shape[0],train.x_shape[1],3),input_tensor = img_conc)    
x = vgg.output
x = Lambda(lambda x : x[:,:,:,0])(x)
x = Flatten()(x)
x = Dense(132,activation="relu")(x)
x = Dropout(0.25)(x)
predictions = Dense(11,activation="softmax")(x)

model = Model(inputs= vgg.input, outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\vgg-spectrum\\models-{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train, epochs=100, verbose=1,shuffle=True, validation_data=valid,callbacks=[checkpoint])