import numpy as np
import os
import json
import keras
from keras.utils import Sequence
from keras.utils import to_categorical
from random import shuffle
import librosa

class WavSpec(Sequence):
    batch_size = 500
    wav_shape = 0
    spec_shape = 0
    x_shape = 0
    wav_dir = ""
    spec_dir = ""

    def __init__(self,spectrum_directory,wav_directory,labels_directory,batch_size = 100):
        file = open(labels_directory + "\\instrument_families","r")
        self.families = json.load(file)
        self.filenames = list(self.families.keys())
        shuffle(self.filenames)
        self.wav_dir = wav_directory
        self.spec_dir = spectrum_directory
        wav,_ = librosa.load(self.wav_dir + self.filenames[0],mono=True,sr=None)
        self.wav_shape = wav.shape
        filename,_,_ = self.filenames[0].partition(".wav")
        spec = np.load(self.spec_dir + filename + ".npy")
        self.spec_shape = spec.shape
        self.batch_size = batch_size
        x = np.concatenate((np.reshape(spec,(-1,)), np.reshape(wav,(-1,))))
        self.x_shape = x.shape
        print("wav shape: " + str(self.wav_shape) + " spec shape " + str(self.spec_shape))
        print("x shape: " + str(self.x_shape))
    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))
    
    def __getitem__(self, index):
        i = 0
        while i < self.batch_size:
            filename,_,_ = self.filenames[index].partition(".wav")
            if i == 0:
                spec_x = np.load(self.spec_dir + filename + ".npy")
                wav_x,_ = librosa.load(self.wav_dir + filename + ".wav",mono=True,sr=None)
                batch_x = np.concatenate((np.reshape(spec_x,(-1,)), np.reshape(wav_x,(-1))))
                batch_y = self.families[self.filenames[index]]
                self.x_shape = batch_x.shape
                if wav_x.shape != self.wav_shape:
                    print("wav shape is different in file " + filename)
            spec_x = np.load(self.spec_dir + filename + '.npy')
            wav_x,_ = librosa.load(self.wav_dir+filename+'.wav',mono=True,sr=None)
            x = np.concatenate((np.reshape(spec_x,(-1,)), np.reshape(wav_x,(-1,))))
            batch_x = np.vstack((batch_x,x))
            batch_y =np.append(batch_y,self.families[self.filenames[index]])
            i +=1
        batch_y = to_categorical(np.array(batch_y),num_classes=11)
        print("in get Item: " +str(i))
        print("wav shape: " + str(self.wav_shape) + " spec shape " + str(self.spec_shape))
        print("x shape: " + str(self.x_shape))
        return batch_x,batch_y