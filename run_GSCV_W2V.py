import sys
sys.path.append("../")
import sys, getopt

import random
import importlib

from transformers.trainer_callback import TrainerControl, TrainerState
from models.Model_Wav2Vec import DataCollatorCTCWithPadding
from models.Model_Wav2Vec import Wav2Vec2ForSpeechClassification
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

import torchaudio
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Processor, AutoConfig, TrainerCallback
from datasets import load_dataset
import os

import torch
import numpy as np
from config import *
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from config import *
from utils import emotion_mapping_wav2vec
from sklearn.model_selection import StratifiedKFold

from dataset.RAVDESSpeechDataset import RAVDESSpeechDataset
from dataset.RAVDESSongDataset import RAVDESSongDataset

from utils import calculate_avg_metrics
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from torchinfo import summary

import pandas as pd
from itertools import product
import multiprocessing
import csv


class GridSearchCV_Wav2Vec():
    def __init__(self, cv, hyperparameters_grid, module_name, model_name, datasets):

        self.model_name = model_name

        self.module = importlib.import_module(module_name)
        self.model_class = getattr(self.module, model_name)
        self.callbacks = getattr(self.module, 'MyCallbacks') 
        #self.trainer_class = getattr(module, 'Trainer')
        #self.processor_class = getattr(module, 'Processor')

        self.hyperparameters_grid = hyperparameters_grid
        self.datasets = datasets
        self.cv = cv

        self.results_df = []
        self.df_pred_gt_list = []
        self.best_avg_accuracy = float('-inf')

        print('\n'+datasets+'\n')

        self.save_model = datasets
        self.path_wav2vec_results = '../../CV_Results/Wav2Vec/'
        self.csv_filename = "../../CV_Results/Wav2Vec/Best_hyperparameters_results.csv"

    def prepare_dataframe_for_training(self, dataset):
        dataset_wav2Vec = []
        for i in range(len(dataset)):
            try:
                tmp = dataset.iloc[i]
                # Check whether there are some broken files
                torchaudio.load(tmp['Path'])
                emotion = emotion_mapping_wav2vec(tmp['Emotion'])
                if emotion != -1:
                    dataset_wav2Vec.append({'Path': str(tmp['Path']), 'Emotion': str(emotion)})
            except Exception as e:
                print(str(tmp['Path']), e)
                pass

        return pd.DataFrame(dataset_wav2Vec)

    def train(self, param_dict, train_df, eval_df, test_df, mycallbacks, result_queue):
        model = self.model_class(param_dict, mycallbacks)
        model.fit(train_df, eval_df, test_df)
        y_pred = model.get_y_pred()
        y_gt = model.get_y_gt()

        result_queue.put([mycallbacks.get_cv_scores_accuracy(), y_pred, y_gt])

    def fit(self):
        ravSpeech = RAVDESSpeechDataset(args=RAVDESSpeechArgs())
        dataset_ravSpeech = ravSpeech.get_dataset()
        ravSong = RAVDESSongDataset(args=RAVDESSongArgs())
        dataset_ravSong = ravSong.get_dataset()
        dataset = pd.concat([dataset_ravSpeech, dataset_ravSong], ignore_index=True)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for param_id, params in enumerate(product(*self.hyperparameters_grid.values())):
            param_dict = dict(zip(self.hyperparameters_grid.keys(), params))
            print(param_dict)

            cv_scores_accuracy=[]
            y_pred_list = []
            y_gt_list = []
            # CV among datasets 
            for train_index, test_index in self.cv.split(dataset, dataset['Emotion']):
                KF_x_train = self.prepare_dataframe_for_training(dataset.iloc[train_index])
                KF_x_test = self.prepare_dataframe_for_training(dataset.iloc[test_index])
                KF_x_train, KF_x_val, _, _ = train_test_split(KF_x_train, KF_x_train['Emotion'], stratify=KF_x_train['Emotion'], test_size=0.1, random_state=42)

                print(KF_x_train['Emotion'].value_counts())
                print(KF_x_val['Emotion'].value_counts())
                print(KF_x_test['Emotion'].value_counts())

                mycallbacks = self.callbacks()
                manager = multiprocessing.Manager()
                result_queue = manager.Queue()
                process = multiprocessing.Process(target=self.train, args=(param_dict, KF_x_train, KF_x_val, KF_x_test, mycallbacks, result_queue))
                process.start()
                process.join()

                tmp_queue = result_queue.get()
                accuracy = tmp_queue[0]
                y_pred = tmp_queue[1]
                y_gt = tmp_queue[2]

                print("Accuracy on Test set: ", accuracy)
                cv_scores_accuracy.append(accuracy)

                y_pred_list.append(y_pred)
                y_gt_list.append(y_gt)
                 
            # Calculate and print the mean and standard deviation of the cross-validation scores
            mean_accuracy = np.mean(cv_scores_accuracy)
            std_accuracy = np.std(cv_scores_accuracy)
            print(f'Mean Accuracy: {mean_accuracy:.4f}')
            print(f'Accuracy Standard Deviation: {std_accuracy:.4f}')

            avg_metrics = calculate_avg_metrics(y_pred_list, y_gt_list)
            param_dict['avg_test_accuracy_returned_by_model'] = mean_accuracy
            param_dict['std_test_accuracy_returned_by_model'] = std_accuracy
            param_dict['avg_test_accuracy_postprocess'] = avg_metrics['avg_accuracy']
            param_dict['avg_precision_postprocess'] = avg_metrics['avg_precision']
            param_dict['avg_recall_postprocess'] = avg_metrics['avg_recall']
            param_dict['avg_f1_postprocess'] = avg_metrics['avg_f1']

            if self.best_avg_accuracy < mean_accuracy:
                self.best_avg_accuracy = mean_accuracy
                self.best_params =param_dict

            # Append the new experiment to the DataFrame
            self.results_df.append(param_dict)
            self.df_pred_gt_list.append({'param_id': param_id, 'y_pred_list':y_pred_list, 'y_gt_list':y_gt_list})

            df1 = pd.DataFrame(self.results_df)
            df2 = pd.DataFrame(self.df_pred_gt_list)

            # Save the updated DataFrame to a CSV file
            # Check if directory exists
            if not os.path.exists(self.path_wav2vec_results):
                # If directory does not exist, create it
                os.makedirs(self.path_wav2vec_results)

            df1.to_csv(os.path.join(self.path_wav2vec_results, f"{self.save_model}_hyperparameters_results.csv"), index=False)
            df2.to_pickle(os.path.join(self.path_wav2vec_results, f"{self.save_model}_pred_gt_lists.pkl"))
        
        # Check if the file exists, and create it if it doesn't
        file_exists = os.path.isfile(self.csv_filename)
        # Open the CSV file in append mode and write the row
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = self.best_params.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file doesn't exist, write the header row first
            if not file_exists:
                writer.writeheader()

            # Write the new row to the CSV file
            writer.writerow(self.best_params)


def main(argv):
    what = 0 # 0: Estimate signals, 1: Perform results evalution, 2: Print metrics, 3: Train models on datasets

    opts, args = getopt.getopt(argv,"ha:", ["action="])
    for opt, arg in opts:
        if opt == '-h':
            print ('run_all.py -a <action>')
            sys.exit()
        elif opt in ("-a", "--action"):
            try:
                what = int(arg)
            except ValueError:
                print('The action parameter must be an integer.')
                sys.exit(2)

    print ('\nAction is ', what)

    
    if what == 0:

        model_name = 'Wav2Vec_Model'
        
        # Specify which module contains the model
        module_name = 'clifford_w2v'

        # Define the hyperparameters grids

        #       x       =   x_0 + x1e1 + x2e2 + x12e12
        # 'blades_idxs' :   0     1      2      3 

        #       x       =   x_0 + x1e1 + x2e2 + x3e3 + x12e12 + x13e13 + x23e23 + x123e123
        # 'blades_idxs' :   0     1      2      3      4        5        6        7

        param_grid = {
            'pooling_mode': ["mean","max"],
            'dense_1_out': [100, 200], 
            'dropout_1': [0.3, 0.5],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0]]
        }

        datasets = 'RAVDESS'
        grid_search_wav2vec = GridSearchCV_Wav2Vec(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), hyperparameters_grid=param_grid, module_name=module_name, model_name=model_name, datasets=datasets)
        grid_search_wav2vec.fit()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main(sys.argv[1:])

multiprocessing.set_start_method('spawn', force=True)
process = multiprocessing.Process(target=main)
process.start()