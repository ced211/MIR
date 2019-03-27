import json
import numpy as np
import os
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
from shutil import copyfile

if __name__ == "__main__":

    #count occurence of each instrument
    instr_occurence = np.zeros(1006,)
    f = open("examples.json", "r")
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
    input = open("examples.json", "r")
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
    target_dir = "C:\\Users\\cedri\\Documents\\nsynth-train\\training_set\\"
    label = {}
    map = {}
    while count < 10 and i < rank.shape[0]:
        id = rank[i]
        print(id)
        if instrument[id][0][1] not in instr_family:
            instr_family.add(instrument[id][0][1])
            for file in instrument[id]:
                src = "C:\\Users\\cedri\\Documents\\nsynth-train\\audio\\" + file[0] +".wav"
                dst =target_dir + file[0] + ".wav"               
                copyfile(src,dst)
                label[file[0]+".wav"] = count
                map[count] = id
            count += 1
        i += 1
    json.dump(label,open("labels","w"))
    json.dump(map,open("labels_to_instr_id","w"))
    
"""
    #process .wav to STFT
    plt.axis('off')
    maxi = 0
    i = 0
    directory = "C:\\Users\\cedri\\Documents\\nsynth-train\\audio"

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            print(filename)
            fs,x = scipy.io.wavfile.read(directory + "\\" + filename)           
            plt.specgram(x,Fs=fs,NFFT = 1024)
            file,_,_ = filename.partition(".wav")
            plt.savefig('C:\\Users\\cedri\\Documents\\nsynth-train\\specgram\\' + file + ".png")
        i += 1
    print(maxi)
"""

