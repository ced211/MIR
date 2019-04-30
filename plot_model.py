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

if __name__ == "__main__":

	#the spectrum model
	test = Mfcc("..\\spectrum-test\\","..\\nsynth-test")
	model = load_model("../models/conv-spectrum/models-02.hdf5")
	plot_model(model, to_file='conv-spectrum.png')

	#the mfcc model
	model = load_model("../models/conv-mfccs/models-02.hdf5")
	plot_model(model, to_file='conv-mfccs.png')

	#the dense-mfccs model
	model = load_model("../models/dense-mfccs/dense-models-02.hdf5")
	plot_model(model, to_file='dense-mfccs.png')

	#the vgg model
	model = load_model("../models/vgg-spectrum/models-02.hdf5")
	plot_model(model, to_file='vgg-spectrum.png')

	#the fq model
	test = FqMfccSeq("..\\spectrum-test\\","..\\nsynth-test")
	model = load_model("../models/conv-spectrum-fq/models-02.hdf5")
	plot_model(model, to_file='conv-spectrum-fq.png')

	#the time model
	test = TimeMfccSeq("..\\spectrum-test\\","..\\nsynth-test")
	model = load_model("../models/conv-spectrum-time/models-02.hdf5")
	plot_model(model, to_file='conv-spectrum-time.png')
	
	#the fq and time model
	test = Seq_time_freq("..\\spectrum-test\\","..\\nsynth-test")
	model = load_model("../models/time_freq-spectrum/models-02.hdf5")
	plot_model(model, to_file='time_freq-spectrum.png')