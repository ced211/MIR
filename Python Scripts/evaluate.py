import numpy as np
from keras.models import load_model
import os

model = load_model("../models/conv-spectrum/models-40.hdf5")
directory = "../spectrum-test/"
test_set = []
for filename in os.listdir(directory):
    test_set.append(np.load(filename))
    name,_,_ = filename.split(".")
    

