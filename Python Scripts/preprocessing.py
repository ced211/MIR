#website
#https://medium.com/swlh/how-to-run-gpu-accelerated-signal-processing-in-tensorflow-13e1633f4bfb?fbclid=IwAR0J054kiuD2atsKxYk4S_ygwk-e3Ma--U4iFZgMuSf2RUUiLbJhb9BmW9A

import numpy as np
import scipy.io.wavfile
import librosa
import os

def wav2mfcc(file_path, max_pad_len=50):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=sr)
    pad_width = max_pad_len - mfcc.shape[1]
    print(pad_width)
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def wav2spectrum(file_path, max_pad_len=50):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]  
    spectrum = librosa.feature.melspectrogram(wave,sr=sr)
    return spectrum

if __name__ == "__main__":
    directory = "../training_set"

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            input_file = directory + "\\" + filename
            file,_,_ = filename.partition(".wav")
            #mfcc_outpath = "mfccs\\" + file + ".npy"
            #mfccs = wav2mfcc(input_file)
            #np.save(mfcc_outpath,mfccs)
            spectrum_outpath = "..\\spectrum\\" + file + ".npy"
            spectrum = wav2spectrum(input_file)
            np.save(spectrum_outpath,spectrum)