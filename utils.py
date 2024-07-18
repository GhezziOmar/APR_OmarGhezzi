import matplotlib.pyplot as plt 
import numpy as np
from config import *
import librosa 
from sklearn.model_selection import train_test_split

from scipy.signal import butter, lfilter
import soundfile as sf
import resampy
import pandas as pd
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
random.seed(42)

features_list = ['mfcc_vec', 'delta1_mfcc_vec', 'delta2_mfcc_vec', 'stft_vec', 'chroma_vec', 'mel_vec', 'contrast_vec', 'mfcc_img_cnn', 'mel_img_cnn']

def get_mfcc_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    mfcc = librosa.feature.mfcc(y=audio_data_cropped_zeros, sr=sample_rate, n_mfcc=128)
    return np.mean(mfcc.T, axis=0)

def get_delta1_mfcc_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    delta1_mfcc = librosa.feature.delta(librosa.feature.mfcc(y=audio_data_cropped_zeros, sr=sample_rate, n_mfcc=128))
    return np.mean(delta1_mfcc.T, axis=0)

def get_delta2_mfcc_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    delta2_mfcc = librosa.feature.delta(librosa.feature.mfcc(y=audio_data_cropped_zeros, sr=sample_rate, n_mfcc=128), order=2)
    return np.mean(delta2_mfcc.T, axis=0)

def get_mel_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    signal = librosa.feature.melspectrogram(y=audio_data_cropped_zeros, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    return np.mean(mel.T, axis=0)

def get_stft_vec(audio_data):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    stft = np.abs(librosa.stft(audio_data_cropped_zeros))
    return np.mean(stft.T, axis=0)

def get_chroma_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    chroma = librosa.feature.chroma_stft(S=np.abs(librosa.stft(audio_data_cropped_zeros)), sr=sample_rate)
    return np.mean(chroma.T, axis=0)

def get_contrast_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(audio_data_cropped_zeros)), sr=sample_rate)
    return np.mean(contrast.T, axis=0)

def get_tonnetz_vec(audio_data, sample_rate):
    audio_data_cropped_zeros = crop_zeros(audio_data)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data_cropped_zeros), sr=sample_rate)
    return np.mean(tonnetz.T, axis=0)

def get_mfcc_img_cnn(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = audio_data, sr=sample_rate, n_mfcc=128)
    return np.array(signal)

def get_mfcc_img_cnn_delta1(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = audio_data, sr=sample_rate, n_mfcc=128)
    delta1 = librosa.feature.delta(signal, order=1)
    return np.array(delta1)

def get_mfcc_img_cnn_delta2(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = audio_data, sr=sample_rate, n_mfcc=128)
    delta2 = librosa.feature.delta(signal, order=2)
    return np.array(delta2)

def get_mfcc_img_cnn_EMD(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = IMFs_groups, sr=sample_rate, n_mfcc=128)
    return np.array(signal)

def get_mfcc_img_cnn_EMD_delta1(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = IMFs_groups, sr=sample_rate, n_mfcc=128)
    delta1 = librosa.feature.delta(signal, order=1)
    return np.array(delta1)

def get_mfcc_img_cnn_EMD_delta2(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.mfcc(y = IMFs_groups, sr=sample_rate, n_mfcc=128)
    delta2 = librosa.feature.delta(signal, order=2)
    return np.array(delta2)

def get_mel_img_cnn(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    return np.array(mel)

def get_mel_img_cnn_delta1(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    delta1 = librosa.feature.delta(mel, order=1)
    return np.array(delta1)

def get_mel_img_cnn_delta2(audio_data, sample_rate):
    if len(audio_data) < 6 * sample_rate:
        audio_data = np.pad(audio_data, pad_width=(0, 6 * sample_rate - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    delta2 = librosa.feature.delta(mel, order=2)
    return np.array(delta2)

def get_mel_img_cnn_EMD(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=IMFs_groups, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    return np.array(mel)

def get_mel_img_cnn_EMD_delta1(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=IMFs_groups, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    delta1 = librosa.feature.delta(mel, order=1)
    return np.array(delta1)

def get_mel_img_cnn_EMD_delta2(IMFs_groups, sample_rate):
    if IMFs_groups.shape[1] < 6 * sample_rate:
        IMFs_groups = np.pad(IMFs_groups, pad_width=((0, 0), (0, 6 * sample_rate - IMFs_groups.shape[1])), mode='constant')
    else:
        IMFs_groups = IMFs_groups[:6 * sample_rate]
    signal = librosa.feature.melspectrogram(y=IMFs_groups, sr=sample_rate, n_mels=128) 
    mel = librosa.power_to_db(signal, ref=np.min)
    delta2 = librosa.feature.delta(mel, order=2)
    return np.array(delta2)

def add_feature(audio_data, sample_rate, feature_name):
    if feature_name == 'mfcc_vec':
        return get_mfcc_vec(audio_data, sample_rate)
    elif feature_name == 'delta1_mfcc_vec':
        return get_delta1_mfcc_vec(audio_data, sample_rate)
    elif feature_name == 'delta2_mfcc_vec':
        return get_delta2_mfcc_vec(audio_data, sample_rate)
    elif feature_name == 'stft_vec':
        return get_stft_vec(audio_data)
    elif feature_name == 'chroma_vec':
        return get_chroma_vec(audio_data, sample_rate)
    elif feature_name == 'mel_vec':
        return get_mel_vec(audio_data, sample_rate)
    elif feature_name == 'contrast_vec':
        return get_contrast_vec(audio_data, sample_rate)
    elif feature_name == 'tonnetz_vec':
        return get_tonnetz_vec(audio_data, sample_rate)
    elif feature_name == 'mfcc_img_cnn':
        return get_mfcc_img_cnn(audio_data, sample_rate)
    elif feature_name == 'delta1_mfcc_img_cnn':
        return get_mfcc_img_cnn_delta1(audio_data, sample_rate)
    elif feature_name == 'delta2_mfcc_img_cnn':
        return get_mfcc_img_cnn_delta2(audio_data, sample_rate)
    elif feature_name == 'mel_img_cnn':
        return get_mel_img_cnn(audio_data, sample_rate)
    elif feature_name == 'delta1_mel_img_cnn':
        return get_mel_img_cnn_delta1(audio_data, sample_rate)
    elif feature_name == 'delta2_mel_img_cnn':
        return get_mel_img_cnn_delta2(audio_data, sample_rate)

def add_feature_EMD(IMFs_groups, sample_rate, feature_name):
    if feature_name == 'mfcc_vec':
        return get_mfcc_vec(IMFs_groups, sample_rate)
    elif feature_name == 'delta1_mfcc_vec':
        return get_delta1_mfcc_vec(IMFs_groups, sample_rate)
    elif feature_name == 'delta2_mfcc_vec':
        return get_delta2_mfcc_vec(IMFs_groups, sample_rate)
    elif feature_name == 'stft_vec':
        return get_stft_vec(IMFs_groups)
    elif feature_name == 'chroma_vec':
        return get_chroma_vec(IMFs_groups, sample_rate)
    elif feature_name == 'mel_vec':
        return get_mel_vec(IMFs_groups, sample_rate)
    elif feature_name == 'contrast_vec':
        return get_contrast_vec(IMFs_groups, sample_rate)
    elif feature_name == 'tonnetz_vec':
        return get_tonnetz_vec(IMFs_groups, sample_rate)
    elif feature_name == 'mfcc_img_cnn':
        return get_mfcc_img_cnn_EMD(IMFs_groups, sample_rate)
    elif feature_name == 'delta1_mfcc_img_cnn':
        return get_mfcc_img_cnn_EMD_delta1(IMFs_groups, sample_rate)
    elif feature_name == 'delta2_mfcc_img_cnn':
        return get_mfcc_img_cnn_EMD_delta2(IMFs_groups, sample_rate)
    elif feature_name == 'mel_img_cnn':
        return get_mel_img_cnn_EMD(IMFs_groups, sample_rate)
    elif feature_name == 'delta1_mel_img_cnn':
        return get_mel_img_cnn_EMD_delta1(IMFs_groups, sample_rate)
    elif feature_name == 'delta2_mel_img_cnn':
        return get_mel_img_cnn_EMD_delta2(IMFs_groups, sample_rate)

def show_spectogram(spectogram, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectogram, sr=sr, x_axis='time')
    plt.title(title)

    plt.colorbar(format='%+2.0f dB')
    plt.show()

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveform {}'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def one_hot_encoding_labels(input_string):
    num_categories = 6
    category_index = int(input_string) - 1  
    
    one_hot_vector = np.zeros(num_categories)
    one_hot_vector[category_index] = 1
    
    return one_hot_vector

def emotion_distribution(dataframe, labels):
        frequency_dict = {}
        for item in dataframe['Emotion']:
            if item in frequency_dict:
                frequency_dict[item] += 1
            else:
                frequency_dict[item] = 1
        
        for k,v in frequency_dict.items():
            print(f'{k}-{labels[k]}: {v} ({round((v/len(dataframe))*100, 2)}%)')

def crop_zeros(signal):
   # Crop leading and trailing zeros from an audio signal.
    nonzero_indices = np.where(signal >= 0.001)[0]

    if len(nonzero_indices) == 0:
        raise ValueError("The input signal is all zeros.")

    start_index = nonzero_indices[0]
    end_index = nonzero_indices[-1]

    cropped_signal = signal[start_index:end_index+1]

    return cropped_signal
  
def filtering_audio(audio_file_path, cutoff_frequency=4000, filter_order=8, target_sample_rate=16000):
    def butter_lowpass(cutoff_freq, sample_rate, order=4):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_butterworth_filter(data, b, a):
        return lfilter(b, a, data)

    audio_data, sample_rate = sf.read(audio_file_path, dtype='int16')
    assert audio_data.dtype == np.int16, 'Bad sample type: %r' % audio_data.dtype
    audio_data = audio_data / 32768.0 # normalize into [-1, +1]
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)

    # Design the Butterworth low-pass filter
    b, a = butter_lowpass(cutoff_frequency, target_sample_rate, order=filter_order)

    # Apply the filter to the audio data
    filtered_audio = apply_butterworth_filter(audio_data, b, a)

    return filtered_audio, target_sample_rate

def split_dataset(dataframe, features_list=None, test_size=0.2):
        feature_dataset_train = []
        feature_dataset_test = []
        label_dataset_train = []
        label_dataset_test = []
        train_dataset, test_dataset = train_test_split(dataframe, test_size=test_size, random_state=seed)

        print("The number of samples in the dataset is %d" % len(dataframe))
        print("The number of samples in the training set is %d" % len(train_dataset))
        print("The number of samples in the test set is %d\n" % len(test_dataset))

        for dat_i, dataset in enumerate([train_dataset, test_dataset]):
            for _, row in dataset.iterrows():
                tmp = np.array([])
                emotion = emotion_mapping(row['Emotion'])
                if emotion != -1:
                    if features_list is not None:
                        if 'mfcc_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['mfcc_vec']
                            else:
                                tmp = np.hstack((tmp, row['mfcc_vec']))
                        if 'delta1_mfcc_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['delta1_mfcc_vec']
                            else:
                                tmp = np.hstack((tmp, row['delta1_mfcc_vec']))
                        if 'delta2_mfcc_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['delta2_mfcc_vec']
                            else:
                                tmp = np.hstack((tmp, row['delta2_mfcc_vec']))
                        if 'stft_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['stft_vec']
                            else:
                                tmp = np.hstack((tmp, row['stft_vec']))
                        if 'chroma_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['chroma_vec']
                            else:
                                tmp = np.hstack((tmp, row['chroma_vec']))
                        if 'mel_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['mel_vec']
                            else:
                                tmp = np.hstack((tmp, row['mel_vec']))
                        if 'contrast_vec' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['contrast_vec']
                            else:
                                tmp = np.hstack((tmp, row['contrast_vec']))
                        if 'mfcc_img_cnn' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['mfcc_img_cnn']
                            else:
                                tmp = np.hstack((tmp, row['mfcc_img_cnn']))
                        if 'mel_img_cnn' in features_list['Features']:
                            if tmp.size == 0 or np.all(tmp == 0):
                                tmp = row['mel_img_cnn']
                            else:
                                tmp = np.hstack((tmp, row['mel_img_cnn']))

                        if dat_i==0:
                            feature_dataset_train.append(tmp)
                            label_dataset_train.append(one_hot_encoding_labels(emotion))
                        else:
                            feature_dataset_test.append(tmp)
                            label_dataset_test.append(one_hot_encoding_labels(emotion))
                    else:
                        if dat_i==0:
                            feature_dataset_train.append(row['Path'])
                            label_dataset_train.append(one_hot_encoding_labels(emotion))
                        else:
                            feature_dataset_test.append(row['Path'])
                            label_dataset_test.append(one_hot_encoding_labels(emotion))

        return np.array(feature_dataset_train), np.array(feature_dataset_test), np.array(label_dataset_train), np.array(label_dataset_test)

def from_dataset_to_array(df, features_list):
    feature_dataset = []
    label_dataset = []
    for _, row in df.iterrows():
        tmp = np.array([])
        emotion = emotion_mapping(row['Emotion'])
        if emotion != -1:
            if 'mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['mfcc_vec']))
            if 'delta1_mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['delta1_mfcc_vec']))
            if 'delta2_mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['delta2_mfcc_vec']))
            if 'stft_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['stft_vec']
                else:
                    tmp = np.hstack((tmp, row['stft_vec']))
            if 'chroma_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['chroma_vec']
                else:
                    tmp = np.hstack((tmp, row['chroma_vec']))
            if 'mel_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mel_vec']
                else:
                    tmp = np.hstack((tmp, row['mel_vec']))
            if 'contrast_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['contrast_vec']
                else:
                    tmp = np.hstack((tmp, row['contrast_vec']))
            if 'mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['mfcc_img_cnn']))
            if 'delta1_mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta1_mfcc_img_cnn']))
            if 'delta2_mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta2_mfcc_img_cnn']))
            if 'mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['mel_img_cnn']))
            if 'delta1_mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta1_mel_img_cnn']))
            if 'delta2_mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta2_mel_img_cnn']))

            feature_dataset.append(tmp)
            label_dataset.append(one_hot_encoding_labels(emotion))

    return np.array(feature_dataset), np.array(label_dataset)

def from_dataset_to_array_EMD(df, features_list):
    feature_dataset = []
    label_dataset = []
    for _, row in df.iterrows():
        tmp = np.array([])
        emotion = emotion_mapping(row['Emotion'])
        if emotion != -1:
            if 'mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['mfcc_vec']))
            if 'delta1_mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['delta1_mfcc_vec']))
            if 'delta2_mfcc_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mfcc_vec']
                else:
                    tmp = np.hstack((tmp, row['delta2_mfcc_vec']))
            if 'stft_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['stft_vec']
                else:
                    tmp = np.hstack((tmp, row['stft_vec']))
            if 'chroma_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['chroma_vec']
                else:
                    tmp = np.hstack((tmp, row['chroma_vec']))
            if 'mel_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mel_vec']
                else:
                    tmp = np.hstack((tmp, row['mel_vec']))
            if 'contrast_vec' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['contrast_vec']
                else:
                    tmp = np.hstack((tmp, row['contrast_vec']))
            if 'mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['mfcc_img_cnn']))
            if 'delta1_mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta1_mfcc_img_cnn']))
            if 'delta2_mfcc_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mfcc_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta2_mfcc_img_cnn']))
            if 'mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['mel_img_cnn']))
            if 'delta1_mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta1_mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta1_mel_img_cnn']))
            if 'delta2_mel_img_cnn' in features_list['Features']:
                if tmp.size == 0 or np.all(tmp == 0):
                    tmp = row['delta2_mel_img_cnn']
                else:
                    tmp = np.hstack((tmp, row['delta2_mel_img_cnn']))

            feature_dataset.append(tmp)
            label_dataset.append(one_hot_encoding_labels(emotion))

    return feature_dataset, label_dataset

def add_zero_dimension(row):
    if len(np.where(np.array(a.shape) == 4)[0]) == 0:
        idx = int(np.where(np.array(a.shape) < 4)[0])
        shape = list(row.shape)
        shape[idx] += 1  # Aumenta la dimensione specificata di 1
        new_array = np.zeros(shape, dtype=row.dtype)
        
        # Crea slices per copiare i dati originali nel nuovo array
        slices = [slice(None)] * new_array.ndim
        for i in range(row.shape[idx]):
            slices[idx] = slice(i, i+1)
            new_array[tuple(slices)] = row.take(i, axis=idx)
        
        new_array[tuple(slices)] = row
    return row
        

def prepare_dataframe_for_training(dataframe):
    condition = dataframe['Emotion'].apply(emotion_mapping) != -1
    filtered_dataset = dataframe[condition].copy()

    return filtered_dataset

def emotion_mapping(emotion_dataset):
    if emotion_dataset == "01" or emotion_dataset == "NEU" or emotion_dataset == "neutral" or emotion_dataset == "n": # neutral 
        return selected_emotion['Neutral']
    elif emotion_dataset == "03" or emotion_dataset == "HAP" or emotion_dataset == "happy" or emotion_dataset == "h": # happy
        return selected_emotion['Happy']
    elif emotion_dataset == "04" or emotion_dataset == "SAD" or emotion_dataset == "sad" or emotion_dataset == "s": # sad
        return selected_emotion['Sad']
    elif emotion_dataset == "05" or emotion_dataset == "ANG" or emotion_dataset == "angry" or emotion_dataset == "a": # angry
        return selected_emotion['Anger']
    elif emotion_dataset == "06" or emotion_dataset == "FEA" or emotion_dataset == "fear" or emotion_dataset == "f": # fear
        return selected_emotion['Fear']
    elif emotion_dataset == "07" or emotion_dataset == "DIS" or emotion_dataset == "disgust" or emotion_dataset == "d": # disgust
        return selected_emotion['Disgust']
    return -1

def emotion_mapping_wav2vec(emotion_dataset):
    if emotion_dataset == "01" or emotion_dataset == "NEU" or emotion_dataset == "neutral" or emotion_dataset == "n": # neutral 
        return 'Neutral'
    elif emotion_dataset == "03" or emotion_dataset == "HAP" or emotion_dataset == "happy" or emotion_dataset == "h": # happy
        return 'Happy'
    elif emotion_dataset == "04" or emotion_dataset == "SAD" or emotion_dataset == "sad" or emotion_dataset == "s": # sad
        return 'Sad'
    elif emotion_dataset == "05" or emotion_dataset == "ANG" or emotion_dataset == "angry" or emotion_dataset == "a": # angry
        return 'Anger'
    elif emotion_dataset == "06" or emotion_dataset == "FEA" or emotion_dataset == "fear" or emotion_dataset == "f": # fear
        return 'Fear'
    elif emotion_dataset == "07" or emotion_dataset == "DIS" or emotion_dataset == "disgust" or emotion_dataset == "d": # disgust
        return 'Disgust'
    return -1

def calculate_avg_metrics(y_pred_list, y_gt_list):
    # Initialize lists to store metrics for each CV fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Iterate over each fold
    for y_pred, y_gt in zip(y_pred_list, y_gt_list):
        # Calculate metrics for each fold
        if len(np.array(y_pred).shape) == 2:
            y_pred = np.argmax(y_pred, axis=1)
            y_gt = np.argmax(y_gt, axis=1)
        accuracy = accuracy_score(y_gt, y_pred)
        precision = precision_score(y_gt, y_pred, average='weighted')
        recall = recall_score(y_gt, y_pred, average='weighted')
        f1 = f1_score(y_gt, y_pred, average='weighted')

        # Append metrics to lists
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Calculate average metrics
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)


    return {
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
    }

def augment_data(X_df, percentage_augmentation, features_list=None, sample_rate=16000):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(p=0.5),
    ])

    df_augmented_features = []
    for emotion in X_df['Emotion'].unique():
        df_emotion = X_df[X_df['Emotion'] == emotion]
        number_of_augmented_samples_emotion_class = int((len(X_df) * percentage_augmentation)/len(X_df['Emotion'].unique()))
        for _ in range(number_of_augmented_samples_emotion_class+abs(np.max(X_df['Emotion'].value_counts())-len(df_emotion))):
            random_index = np.random.randint(0, len(df_emotion))
            dict_features = df_emotion.iloc[random_index].to_dict()
            dict_features['augmented'] = True
            augmented_samples = augment(samples=dict_features['audio'], sample_rate=sample_rate)
            dict_features['audio'] = augmented_samples
            dict_features['sample_rate'] = sample_rate  
            if features_list is not None:
                for feature in features_list['Features']:
                    dict_features[feature] = add_feature(augmented_samples, sample_rate, feature)
            df_augmented_features.append(dict_features)
    
    return pd.concat([X_df, pd.DataFrame(df_augmented_features)]).sample(frac=1, random_state=42).reset_index(drop=True)

def augment_data_EMD(X_df, percentage_augmentation, features_list=None, sample_rate=16000):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(p=0.5),
    ])

    df_augmented_features = []
    for emotion in X_df['Emotion'].unique():
        df_emotion = X_df[X_df['Emotion'] == emotion]
        number_of_augmented_samples_emotion_class = int((len(X_df) * percentage_augmentation)/len(X_df['Emotion'].unique()))
        for _ in range(number_of_augmented_samples_emotion_class+abs(np.max(X_df['Emotion'].value_counts())-len(df_emotion))):
            random_index = np.random.randint(0, len(df_emotion))
            dict_features = df_emotion.iloc[random_index].to_dict()
            dict_features['augmented'] = True
            augmented_samples = augment(samples=dict_features['IMF_groups'], sample_rate=sample_rate)
            dict_features['IMF_groups'] = augmented_samples
            dict_features['sample_rate'] = sample_rate  
            if features_list is not None:
                for feature in features_list['Features']:
                    dict_features[feature] = add_feature_EMD(augmented_samples, sample_rate, feature)
            df_augmented_features.append(dict_features)
    
    return pd.concat([X_df, pd.DataFrame(df_augmented_features)]).sample(frac=1, random_state=42).reset_index(drop=True)

def convert_signature_to_string(g):
    if isinstance(g[0], int):
        signature = g
        num_positives = sum(1 for elem in signature if elem == 1)
        num_negatives = sum(1 for elem in signature if elem == -1)
        num_zeros = len(signature) - num_positives - num_negatives
        algebra = 'CL' + str(num_positives) + str(num_negatives)
        if num_zeros > 0:
            algebra += '0' * num_zeros
        return algebra
    else:
        algebras = []
        for signature in g:
            num_positives = sum(1 for elem in signature if elem == 1)
            num_negatives = sum(1 for elem in signature if elem == -1)
            num_zeros = len(signature) - num_positives - num_negatives
            algebra = 'CL' + str(num_positives) + str(num_negatives)
            if num_zeros > 0:
                algebra += '0' * num_zeros
            algebras.append(algebra)
        return algebras

def convert_blades_to_string(blades_idxs, num_blades):
    conversion_table = {
        2: ['scalar', 'vec1', 'vec2', 'bivec12'],
        3: ['scalar', 'vec1', 'vec2', 'vec3', 'bivec12', 'bivec13', 'bivec23', 'trivec123']
    }

    if isinstance(blades_idxs, int):
        return conversion_table[num_blades][blades_idxs]

    blade_strings = [conversion_table[num_blades][idx] for idx in blades_idxs]
    return '_'.join(blade_strings)

def visualize_imf(IMF, nIMF, N, Fs, file):
    plt.figure()
    plt.ion()
    for k in range(nIMF):
        plt.subplot(nIMF,1,k+1); plt.plot(np.linspace(0, int(N/Fs), N), IMF[k,:])
        plt.title(r''+str(k+1)+'th IMF')
        plt.xlabel('$t$');
        plt.ylabel(r''+str(k+1)+'th $IMF(t)$')
    plt.title(r'Residual')
    plt.xlabel('$t$');
    plt.ylabel(r'$res(t)$')
    plt.tight_layout()
    plt.savefig(file)


def cluster_images(X, y, pca_components=50, tsne_components=2, n_clusters=6, random_state=42):
    X_reshaped = np.reshape(X, (X.shape[0], -1))  # (samples, height*width*channels)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_reshaped)

    tsne = TSNE(n_components=tsne_components, random_state=random_state)
    X_tsne = tsne.fit_transform(X_pca)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_tsne)

    y_true = np.argmax(y, axis=1)  # Converti one-hot encoding in etichette
    conf_matrix = confusion_matrix(y_true, clusters)
    acc = accuracy_score(y_true, clusters)

    # Visualizzazione
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='viridis')
    plt.title('t-SNE clustering')
    plt.savefig('t-SNE_clustering.pdf')

    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy: ", acc)

    return conf_matrix, acc

# Esempio di utilizzo
# X = ... (il tuo dataset di immagini)
# y = ... (le tue etichette one-hot encoded)
# conf_matrix, acc = cluster_images(X, y)
