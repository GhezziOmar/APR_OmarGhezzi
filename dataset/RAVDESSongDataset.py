import os
import pandas as pd
import numpy as np
from config import *
from utils import * 
import emd

class RAVDESSongDataset():
    """
    Class describing the dataset RAVDESS_SONG.

    Filename example: 02-01-06-01-02-01-12.wav
                      M -V- E -E -S- R -A
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    - Vocal channel (01 = speech, 02 = song).
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    - Emotional intensity (01 = normal, 02 = strong). There is no strong intensity for the 'neutral' emotion.
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    - Repetition (01 = 1st repetition, 02 = 2nd repetition).
    - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """

    def __init__(self, args, EMD=False, features_list=None, verbose=1):
        self.args = args
        self.features_list = features_list
        self.verbose = verbose
        self.EMD = EMD
        self.dataframe = pd.DataFrame(columns=['Modality', 'Vocal_Channel', 'Emotion', 'Emotional_Intensity', 'Statement', 
                                               'Repetition', 'Actor', 'Path', 'sample_rate'])
        
        self.labels_statistics = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

        if self.EMD:
            self.nIMFs = 11
            if os.path.exists(os.path.join(self.args.save_path_dataset_EMD, self.args.dataset_name+'.pkl')) == False:
                self.dataframe = self.load_dataset_EMD()
            else:
                self.dataframe = pd.read_pickle(os.path.join(self.args.save_path_dataset_EMD, self.args.dataset_name+'.pkl'))
                if features_list is None:
                    self.dataframe = self.filter_dataset()
                else:
                    self.dataframe = self.filter_dataset(filt_dataset={'Features': features_list})
        else:
            # load the dataset
            if os.path.exists(os.path.join(self.args.save_path_dataset, self.args.dataset_name+'.pkl')) == False:
                self.dataframe = self.load_dataset()
            else:
                self.dataframe = pd.read_pickle(os.path.join(self.args.save_path_dataset, self.args.dataset_name+'.pkl'))
                if features_list is None:
                    self.dataframe = self.filter_dataset()
                else:
                    self.dataframe = self.filter_dataset(filt_dataset={'Features': features_list})

        # shuffle the DataFrame rows
        self.shuffle()

        if self.verbose:
            self.info()
    
    def info(self):
        """
        Dataset description.
        """
        print(f"Dataset:    {self.args.dataset_name}")
        print(f"num items:  {self.size()}:")
        print("-------")
        emotion_distribution(self.dataframe, self.labels_statistics)
        print("-------")

    def shuffle(self, seed: int = 0):
        # shuffle the DataFrame rows
        self.dataframe = self.dataframe.sample(frac=1, random_state=seed)

    def load_dataset(self):
        data_list = []
        for i, dir in enumerate(os.listdir(self.args.data_path)):
            for i, wav_file in enumerate(os.listdir(os.path.join(self.args.data_path, dir))):
                if wav_file.endswith(".wav"):
                    try:
                        filename = wav_file.split('.')[0]
                        filename = filename.split('-')
                        path = os.path.join(self.args.data_path, dir, wav_file)
                        # add features parsed from filename
                        dict_features = {'Path': path, 'Modality': filename[0], 'Vocal_Channel': filename[1], 'Emotion': filename[2], 'Emotional_Intensity': filename[3], 
                                        'Statement': filename[4], 'Repetition': filename[5], 'Actor': filename[6]}
                        # add specified handcrafted features
                        audio_data, sample_rate = filtering_audio(path)
                        dict_features['audio'] = audio_data
                        dict_features['sample_rate'] = sample_rate  
                        for feature in self.features_list:
                            audio_data_cropped_zeros = crop_zeros(audio_data) # crop zeros at the beginning and at the end of the signal
                            dict_features[feature] = add_feature(audio_data_cropped_zeros, sample_rate, feature)
                        data_list.append(dict_features)
                    except ValueError:
                        print(f"ValueError: The input signal is all zeros. File: {wav_file}")

        self.dataframe = pd.concat([self.dataframe, pd.DataFrame(data_list)])
        self.dataframe.to_pickle(os.path.join(self.args.save_path_dataset, self.args.dataset_name+'.pkl'))

        return self.dataframe
    
    def load_dataset_EMD(self):
        data_list = []
        for i, dir in enumerate(os.listdir(self.args.data_path)):
            for i, wav_file in enumerate(os.listdir(os.path.join(self.args.data_path, dir))):
                if wav_file.endswith(".wav"):
                    try:
                        filename = wav_file.split('.')[0]
                        filename = filename.split('-')
                        path = os.path.join(self.args.data_path, dir, wav_file)
                        # add features parsed from filename
                        dict_features = {'Path': path, 'Modality': filename[0], 'Vocal_Channel': filename[1], 'Emotion': filename[2], 'Emotional_Intensity': filename[3], 
                                        'Statement': filename[4], 'Repetition': filename[5], 'Actor': filename[6]}
                        # add specified handcrafted features
                        audio_data, sample_rate = filtering_audio(path)
                        IMFs = emd.sift.sift(audio_data, max_imfs=self.nIMFs)
                        if IMFs.shape[1]<self.nIMFs:
                            IMF_residuals = np.zeros((self.nIMFs-IMFs.shape[1], IMFs.shape[0])).T
                            IMFs = np.vstack([IMFs.T, IMF_residuals.T]).T
                        #visualize_imf(IMFs.T, self.nIMFs, IMFs.shape[0], sample_rate, 'EMD_'+filename[0]+'.pdf')
                        IMFs_HF = IMFs[:,0]
                        IMFs_MF = IMFs[:,1]+IMFs[:,2]
                        IMFs_LF = IMFs[:,3]+IMFs[:,4]+IMFs[:,5]+IMFs[:,6]
                        IMFs_VLF = IMFs[:,7]+IMFs[:,8]+IMFs[:,9]+IMFs[:,10]
                        IMFs_groups = np.array([IMFs_HF, IMFs_MF, IMFs_LF, IMFs_VLF])
                        #visualize_imf(IMFs_groups, 4, IMFs.shape[0], sample_rate, 'EMD_groups'+filename[0]+'.pdf')
                        dict_features['IMF_groups'] = IMFs_groups
                        dict_features['sample_rate'] = sample_rate 
                        for feature in self.features_list:
                            IMFs_groups_cropped_zeros = crop_zeros(IMFs_groups) # crop zeros at the beginning and at the end of the signal
                            dict_features[feature] = add_feature_EMD(IMFs_groups_cropped_zeros, sample_rate, feature) 
                        data_list.append(dict_features)
                    except ValueError:
                        print(f"ValueError: The input signal is all zeros. File: {wav_file}")

        self.dataframe = pd.concat([self.dataframe, pd.DataFrame(data_list)])
        self.dataframe.to_pickle(os.path.join(self.args.save_path_dataset_EMD, self.args.dataset_name+'.pkl'))

        return self.dataframe
    
    def get_dataset(self):
        return self.dataframe
    
    def get_entry_by_filename(self, filename):
        return self.dataframe[self.dataframe['Path'] == os.path.join(self.args.data_path, filename)]
    
    def get_entry_by_index(self, index):
        return self.dataframe.iloc[index]
    
    def size(self) -> int:
        return len(self.dataframe)

    def filter_dataset(self, filt_dataset={'Features': []}):
        tmp = self.dataframe
        if 'Features' in filt_dataset.keys():
            tmp = self.dataframe.drop(columns=[x for x in features_list if x not in filt_dataset['Features']])
        elif 'Modality' in filt_dataset.keys():
            tmp = tmp[tmp['Modality'].isin(filt_dataset['Modality'])]
        elif 'Vocal_Channel' in filt_dataset.keys():
            tmp = tmp[tmp['Vocal_Channel'].isin(filt_dataset['Vocal_Channel'])]
        elif 'Emotion' in filt_dataset.keys():
            tmp = tmp[tmp['Emotion'].isin(filt_dataset['Emotion'])]
        elif 'Emotional_Intensity' in filt_dataset.keys():
            tmp = tmp[tmp['Emotional_Intensity'].isin(filt_dataset['Emotional_Intensity'])]
        elif 'Statement' in filt_dataset.keys():
            tmp = tmp[tmp['Statement'].isin(filt_dataset['Statement'])]
        elif 'Repetition' in filt_dataset.keys():
            tmp = tmp[tmp['Repetition'].isin(filt_dataset['Repetition'])]    
        elif 'Actor' in filt_dataset.keys():
            tmp = tmp[tmp['Actor'].isin(filt_dataset['Actor'])] 

        return tmp



        


        