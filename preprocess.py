import librosa
import os
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import numpy as np

DATA_PATH = "./audio/"

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def mfcc1(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc into vectors
        mfcc_vect = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = mfcc1(wavfile, max_pad_len=max_pad_len)
            mfcc_vect.append(mfcc)
        np.save(label + '.npy', mfcc_vect)



save_data_to_array(path=DATA_PATH, max_pad_len=11)

print  ("Sucessfully Audio converted into Numpy")
