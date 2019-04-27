import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm

import time
import keras
from keras.utils import to_categorical, plot_model
from keras.models import load_model
from mfcc_seq import Mfcc
from Mfcc_freq_seq import FqMfccSeq
from mfcc_time_seq import TimeMfccSeq
from mfcc_seq_time_freq import Seq_time_freq
import copy
import datetime
import time

if __name__ == "__main__":
	
    #test of the fq and time model
    begin = time.time()
    test = Seq_time_freq("..\\spectrum-test\\","..\\nsynth-test")
    end = time.time()
    print("time for test set",end-begin)
    model = load_model("../models/time_freq-spectrum/models-100.hdf5")
    plot_model(model, to_file='time_freq-spectrum.png')
    begin = time.time()
    [loss,accuracy] = model.evaluate_generator(test)
    end = time.time()
    print("time for test set",end-begin)
    print(model.metrics_names)
    print("model on fq and time: loss and accuracy on the testset:",loss,accuracy)