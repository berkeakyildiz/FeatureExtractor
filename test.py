from joblib import Parallel, delayed
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import multiprocessing
import csv
import os
import librosa
import multiprocessing

import numpy as np
import scipy.misc
import os
import argparse
from tqdm import tqdm


def __extract_hpss_melspec_bf(audio_fpath, saved_path, idOfMusic):
    """
    Extension of :func:`__extract_melspec`.
    Not used as it's about ten times slower, but
    if you have resources, try it out.
    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    print(audio_fpath)
    y, sr = librosa.load(audio_fpath, sr=22050)
    genre = feature_path + str(data[idOfMusic])

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(y) / 512) <= stft.shape[1] <= np.ceil(len(y) / 512) + 1
    del y

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    # print(mel)
    del stft
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    print(mfcc)
    #print(array)
    with open( "features.csv", "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(idOfMusic + "," + mfcc + "," + genre)


def process_input(file):
    try:
        #         print(file)
        idOfMusic = int(file[file.rfind("\\") + 1:].replace(".mp3", ""))
        __extract_hpss_melspec_bf(file, "mfcc\\", idOfMusic)
    except:
        print("HATA")


num_cores = 8

feature_path = "D:\\470 proje\\features\\"
data = {}
with open('train_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if "track_id" not in row:
            data[int(row[0])] = row[1]
dataset = 'D:\\470 proje\\fma_medium\\'
subdirs = [x[0] for x in os.walk(dataset)]
del subdirs[0]
for dirName in subdirs:
    music_dir = str(dirName)
    files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.endswith(".mp3")]
    results = Parallel(n_jobs=num_cores)(delayed(process_input)(file) for file in files)