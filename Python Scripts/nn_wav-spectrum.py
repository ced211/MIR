import numpy as np
from keras.layers import Input, Dense, Add, Lambda,Dropout,Flatten
from keras.models import Model
from Wav_spec_seq import WavSpec
from keras.layers import Dense, Dropout,Reshape,Conv1D,MaxPooling1D
from keras.models import load_model
import keras
import keras.backend as K
if __name__ == "__main__":

    train = WavSpec('../spectrum-train/','../audio/',"../nsynth-train")
    valid = WavSpec('../spectrum-valid/','../../nsynth-valid/audio/',"../nsynth-valid")

    #model on spectrum
    input = Input(train.x_shape)
    spec = train.spec_shape[0] * train.spec_shape[1]
    spec_input = Lambda(lambda y: y[:,:spec], output_shape = (spec,))(input)
    spec_input = Reshape((train.spec_shape[0],train.spec_shape[1],1))(spec_input)
    base_model = load_model("../models/conv-spectrum/models-45.hdf5")
    spectrum_model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_1').output)
    spectrum_model.summary()    
    spec_out = spectrum_model(spec_input)

    
    #model on raw wav audio.
    print("wav")
    wav_input = Lambda(lambda x: x[:,spec:], output_shape = (train.wav_shape[0],)) (input) 
    wav_input = Reshape((train.wav_shape[0],1)) (wav_input)
    debug_model = Model(inputs=input,outputs=wav_input)
    debug_model.summary()     
    wav = Conv1D(16, kernel_size= 4, activation='relu', input_shape=(train.wav_shape[0],1)) (wav_input)
    wav = Dropout(0.25)(wav)
    wav = MaxPooling1D(pool_size=4)(wav)
    wav = Conv1D(16, kernel_size= 4, activation='relu') (wav)
    wav = Dropout(0.25)(wav)
    wav = MaxPooling1D(pool_size=4)(wav)
    wav = Conv1D(16, kernel_size= 4, activation='relu') (wav)
    wav = Dropout(0.25)(wav)
    wav = MaxPooling1D(pool_size=4)(wav)
    wav = Flatten()(wav)
    wav = Dense(128,activation = 'relu')(wav)
    wav = Dropout(0.25)(wav)
    
    #merge both model
    x = keras.layers.concatenate([spec_out, wav])
    x = Dense(128,activation = 'relu')(x)
    predictions = Dense(11,activation = 'softmax')(x)
    model = Model(inputs= input, outputs=predictions)

    #Only trained wav and top layer

    #freeze spectrum weight
    for layer in spectrum_model.layers:
        layer.trainable = False  

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\wav-spectrum\\models-{epoch:02d}_tune_wav" 
        + ".hdf5", monitor='val_loss', 
        verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train, epochs=50, verbose=1, validation_data=valid,callbacks=[checkpoint])

    #fine tune
    for layer in spectrum_model.layers:
        layer.trainable = True

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint("..\\models\\wav-spectrum\\models-{epoch:02d}_fine_tune" 
        + ".hdf5", monitor='val_loss', 
        verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train, epochs=25, verbose=1, validation_data=valid,callbacks=[checkpoint])
     


        

