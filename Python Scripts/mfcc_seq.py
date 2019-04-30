import numpy as np
import os
import json
import keras
from keras.utils import Sequence
from keras.utils import to_categorical
import random

class Mfcc(Sequence):
    batch_size = 1000
    valid = None
    train = None
    def __init__(self,mfcc_directory,labels_directory,batch_size = 1000, is_valid=False,ratio=0.7):
        file = open(labels_directory + "\\instrument_families","r")
        self.families = json.load(file)
        self.filenames = list(self.families.keys())
        self.mfcc_dir = mfcc_directory
        filename,_,_ = self.filenames[0].partition(".wav")
        random.seed(11)
        random.shuffle(self.filenames)
        bound = int(round(len(self.filenames)*ratio))
        self.train = self.filenames[:bound]
        self.valid = self.filenames[bound:]
        x = np.load(self.mfcc_dir + filename + ".npy")
        self.x_shape = x.shape
        self.batch_size = batch_size
        self.isvalid = is_valid

    def __len__(self):
        if self.isvalid:
            return int(np.ceil(len(self.valid) / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.train) / float(self.batch_size)))
    
    def __getitem__(self, index):
        i = 0
        while i < self.batch_size:
            if self.isvalid:
                filename, _, _ = self.valid[index % len(self.valid)].partition(".wav")
            else:
                filename, _, _ = self.train[index % len(self.train)].partition(".wav")
            if i == 0:
                batch_x = np.load(self.mfcc_dir + filename + ".npy")
                batch_y = self.families[self.filenames[index]]
            x = np.load(self.mfcc_dir + filename + '.npy')
            batch_x = np.vstack((batch_x,x))
            batch_y =np.append(batch_y,self.families[self.filenames[index]])
            i +=1
        batch_x = batch_x.reshape((len(batch_y),-1,x.shape[1]))
        batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1)
        batch_y = to_categorical(np.array(batch_y),num_classes=11)

        return batch_x,batch_y
            

