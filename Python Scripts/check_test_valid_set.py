import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm

def check_instrument():
    # train set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-train\\instruments","r")
    instruments = json.load(file)
    i = 0
    k = 1
    for filename, instrument in instruments.items():
        if i == 0:
            inst = instrument
            i = 1
            continue
        if i == 1000*k:
            print("process ", i, " files")
            k += 1
        find = np.where( inst == instrument)
        if find[0].size == 0:
            inst = np.append(inst,instrument)
        i += 1
    nb_error = 0
	# validation set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-valid\\instruments","r")
    instruments = json.load(file)
    i = 0
    k = 1
    for filename, instrument in instruments.items():
        find = np.where( inst == instrument)
        if find[0].size == 0:
            if nb_error == 0:
                inst_error = instrument
                nb_error = 1
                continue
            inst_error = np.append(inst_error,instrument)
            nb_error += 1
        if i == 1000*k:
            print("process ", i, " files")
            k += 1
        i += 1
	
	# test set
    file = open("C:\\Users\\Julie\\Documents\\university\\cours_4e\\2e_quadri\\Deep learning\\project\\nsynth-test\\instruments","r")
    instruments = json.load(file)
    i = 0
    k = 1
    for filename, instrument in instruments.items():
        find = np.where( inst == instrument)
        if find[0].size == 0:
            if nb_error == 0:
                inst_error = instrument
                nb_error = 1
                continue
            inst_error = np.append(inst_error,instrument)
            nb_error += 1
        if i == 1000*k:
            print("process ", i, " files")
            k += 1
        i += 1
    print("error", inst_error)
    print(nb_error)	
		
if __name__ == "__main__":
    check_instrument()