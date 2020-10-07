import librosa
import numpy as np
import librosa.display
import pandas as pd
import os, fnmatch
import os
from scipy import stats
from tqdm import tqdm


import utils

# listOfFiles = os.listdir(path)
# pattern = "*.mp3"
# for entry in listOfFiles:
#     if fnmatch.fnmatch(entry, pattern):
#         name = entry.split('.')
#         name = name[0]
#         print(name)
#         #print (path + entry)
#         y, sr = librosa.load(path + entry)

#         #print(type(y), type(sr))
#         #print(y)

#         D = librosa.stft(y)


#         S_full, phase = librosa.magphase(librosa.stft(y))


#         S_filter = librosa.decompose.nn_filter(S_full,
#                                                aggregate=np.median,
#                                                metric='cosine',
#                                                width=int(librosa.time_to_frames(2, sr=sr)))

#         # The output of the filter shouldn't be greater than the input
#         # if we assume signals are additive.  Taking the pointwise minimium
#         # with the input spectrum forces this.
#         S_filter = np.minimum(S_full, S_filter)
#         margin_i, margin_v = 2, 10
#         power = 2

#         mask_i = librosa.util.softmask(S_filter,
#                                        margin_i * (S_full - S_filter),
#                                        power=power)

#         mask_v = librosa.util.softmask(S_full - S_filter,
#                                        margin_v * S_filter,
#                                        power=power)

#         # Once we have the masks, simply multiply them with the input spectrum
#         # to separate the components

#         S_foreground = mask_v * S_full
#         S_background = mask_i * S_full


#         S_background = librosa.istft(S_background)
#         S_foreground = librosa.istft(S_foreground)

#         D_harmonic, D_percussive = librosa.decompose.hpss(D)
#         D_harmonic = librosa.istft(D_harmonic)
#         D_percussive = librosa.istft(D_percussive)


#         librosa.output.write_wav('../Librosa/music_background/' + name + '_background.wav', S_background, sr)
#         librosa.output.write_wav('../Librosa/music_foreground/' + name + '_foreground.wav', S_foreground, sr)
#         librosa.output.write_wav('../Librosa/music_harmonic/' + name + '_harmonic.wav', D_harmonic, sr)
#         librosa.output.write_wav('../Librosa/music_percussive/' + name + '_percussive.wav', D_percussive, sr)
path = 'D:\\MachineLearning\\fma_medium\\'




def compute_features(entry, i):


    try:
        filepath = path + i + '\\' + entry + '.mp3'
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x


        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        #print(mel)
        del stft
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        print(mfcc)
    except Exception as e:
        print('{}: {}'.format(entry, repr(e)))
#
# def save(features, ndigits, name):
#
#     # Should be done already, just to be sure.
#     features.to_csv('features.csv', float_format='%.{}e'.format(ndigits), mode='a', index=1)
#     name.to_csv('features.csv', float_format='%.{}e'.format(ndigits), mode='a', index=0)
#     #np.savetxt("features.csv", features, delimiter=",")
def main():
    i = '000'
    listOfFiles = os.listdir(path + i)
    pattern = "*.mp3"
    print(path)
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            name = entry.split('.')
            name = name[0]
            #print(entry)

            entries = compute_features(name, i)
            #entries.loc[entry.name] = entry
            print(entries)
            # print(features)
#           name = pd.Series(dtype=np.str)
            # save(entries, 5, name)


if __name__ == "__main__":
    main()
