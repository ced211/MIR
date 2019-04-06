import json
import numpy as np
import os
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
from shutil import copyfile

if __name__ == "__main__":

    #count occurence of each instrument in training set 
    instr_occurence = np.zeros(1006,)
    f = open("C:\\Users\\cedri\\Documents\\nsynth-train\\examples.json", "r")
    for line in f:
        start,delim,end = line.partition("\"instrument\":")
        if delim != "":
            instID,_,_ = end.partition(",")
            instr_occurence[int(instID)] += 1
    #give id of instrument with most data.
    instr_occ_rank = sorted(range(len(instr_occurence)), key=lambda k: instr_occurence[k],reverse=True)
    np.savetxt("inst_occ",instr_occurence)
    np.savetxt("instr_rank",instr_occ_rank)

    #filter json
    input = open("C:\\Users\\cedri\\Documents\\nsynth-train\\examples.json", "r")
    data = json.load(input)
    instrument = {}
    for d in data:
        id = data[d]['instrument']
        if id not in instrument:
            instrument[id] = []
        instrument[id].append((d,data[d]['instrument_family']))

    rank = np.loadtxt("instr_rank")
    instr_family = set()
    count = 0
    i = 0

    #make training set.
    target_dir = "C:\\Users\\cedri\\Documents\\nsynth-train\\training_set\\"
    label = {}
    map = {}
    while count < 10 and i < rank.shape[0]:
        id = rank[i]
        if instrument[id][0][1] not in instr_family:
            instr_family.add(instrument[id][0][1])
            for file in instrument[id]:
                src = "C:\\Users\\cedri\\Documents\\nsynth-train\\audio\\" + file[0] +".wav"
                dst =target_dir + file[0] + ".wav"               
                #copyfile(src,dst)
                label[file[0]+".wav"] = count
                map[count] = id
            count += 1
        i += 1
    inv_map = {v: k for k, v in map.items()}
    json.dump(inv_map,open("C:\\Users\\cedri\\Documents\\nsynth-train\\instr_id_to_labels","w"))
    json.dump(label,open("C:\\Users\\cedri\\Documents\\nsynth-train\\labels","w"))
    json.dump(map,open("C:\\Users\\cedri\\Documents\\nsynth-train\\labels_to_instr_id","w"))

    #make validation set.
    print("making validation set")
    file = open("C:\\Users\\cedri\\Documents\\nsynth-valid\\examples.json", "r")
    examples = json.load(file)
    target_dir = "C:\\Users\\cedri\\Documents\\nsynth-valid\\validation_set\\"
    label = {}
    for key in examples:
        print(key)
        instr_id = examples[key]['instrument']
        #problem, if is never true !
        if instr_id in inv_map:
            print("copy file")
            src = "C:\\Users\\cedri\\Documents\\nsynth-valid\\audio\\" + key + ".wav"
            dst = target_dir + key + ".wav"
            copyfile(src,dst)
            label[key + ".wav"] = inv_map[instr_id]
    json.dump(label,open("C:\\Users\\cedri\\Documents\\nsynth-valid\\labels", "w"))
    print(inv_map)


    

