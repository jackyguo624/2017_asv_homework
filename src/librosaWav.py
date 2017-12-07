import librosa
import os
from tqdm import tqdm
import numpy as np

sample_rate = 16000
extend_num = 5
n_fft = int(25 * sample_rate / 1000)
hop_length = int(10 * sample_rate / 1000)

def get_feats(file_path, labels):
    feats = []
    targets = []
    for root , dirs ,files in  os.walk(file_path):
        for f in tqdm(files):
            idx = f
            f = os.path.join(root, f)
            try:
                audio, _ = librosa.load(f, sr=sample_rate, )
            except:
                print (f)
                continue
            # get 39 dim feature
            mfcc = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            mfcc_delta_1 = librosa.feature.delta(mfcc)
            mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
            feature = np.concatenate((mfcc, mfcc_delta_1, mfcc_delta_2), axis=0)
            feats.append(feature.T)
            targets.append(np.array([int(labels[idx])] * len(feature.T)).T)
    return feats, targets
