import numpy as np
import os
import json
import keras
from keras.utils import Sequence
from keras.utils import to_categorical
from random import shuffle

class Seq_time_freq(Sequence):
    batch_size = 1000

    def __init__(self, mfcc_directory, labels_directory, batch_size=1000):
        file = open(labels_directory + "\\instrument_families", "r")
        self.families = json.load(file)
        self.filenames = list(self.families.keys())
        self.mfcc_dir = mfcc_directory
        filename, _, _ = self.filenames[0].partition(".wav")
        shuffle(self.filenames)
        time = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 0)
        freq = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 1)
        x = np.concatenate((time,freq),axis=None)
        self.x_shape = x.shape
        print(self.x_shape)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        i = 0
        while i < self.batch_size:
            filename, _, _ = self.filenames[index].partition(".wav")
            if i == 0:
                time = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 0)
                freq = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 1)
                x = np.concatenate((time,freq),axis=None)
                batch_x = x
                batch_y = self.families[self.filenames[index]]
            time = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 0)
            freq = np.sum(np.load(self.mfcc_dir + filename + ".npy"), 1)
            x = np.concatenate((time,freq),axis=None)            
            batch_x = np.vstack((batch_x, x))
            batch_y = np.append(batch_y, self.families[self.filenames[index]])
            i += 1
        batch_y = to_categorical(np.array(batch_y), num_classes=11)

        return batch_x, batch_y
