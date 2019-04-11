import json
import numpy as np
import os
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
from shutil import copyfile

if __name__ == "__main__":

    #train set
    input = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\examples.json", "r")
    data = json.load(input)
    instrument_train = {}
    instr_family_train = {}
    for d in data:
        id = data[d]['instrument']
        id_family = data[d]['instrument_family']
        instrument_train[d + ".wav"] = id
        instr_family_train[d + ".wav"] = id_family
    json.dump(instrument_train,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\instruments","w"))
    json.dump(instr_family_train,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\instrument_families","w"))
	
	#validation set
    input = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-valid\\examples.json", "r")
    data = json.load(input)
    instrument_valid = {}
    instr_family_valid = {}
    for d in data:
        id = data[d]['instrument']
        id_family = data[d]['instrument_family']
        instrument_valid[d + ".wav"] = id
        instr_family_valid[d + ".wav"] = id_family
    json.dump(instrument_valid,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-valid\\instruments","w"))
    json.dump(instr_family_valid,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-valid\\instrument_families","w"))
	
	#test set
    input = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-test\\examples.json", "r")
    data = json.load(input)
    instrument_test = {}
    instr_family_test = {}
    for d in data:
        id = data[d]['instrument']
        id_family = data[d]['instrument_family']
        instrument_test[d + ".wav"] = id
        instr_family_test[d + ".wav"] = id_family
    json.dump(instrument_test,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-test\\instruments","w"))
    json.dump(instr_family_test,open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-test\\instrument_families","w"))

    

