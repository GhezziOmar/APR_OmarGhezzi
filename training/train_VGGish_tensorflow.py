import sys
sys.path.append("../")

import os
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from itertools import product
from dataset.RAVDESSpeechDataset import RAVDESSpeechDataset
from dataset.RAVDESSongDataset import RAVDESSongDataset
from models.Model_VGGish import VGGish_Model

from config import *
from utils import *

import multiprocessing

class GridSearchCV:
    def __init__(self, cv, param_grid=None, model_class=None):
        self.cv = cv
        self.hyperparameters_grid = param_grid
        self.model_class = model_class 

        self.best_avg_accuracy = float('-inf')
        self.best_params = None
        self.results_df = []
        self.df_pred_gt_list = []

        self.path_wav2vec_results = '../../CV_Results/VGGish/'
        self.csv_filename = "../../CV_Results/VGGish/Best_hyperparameters_results.csv"

    def check_disjoint(self, list1, list2, list3):
        # Convert lists to sets
        set1 = set(list1)
        set2 = set(list2)
        set3 = set(list3)  

        # Find intersections
        common_elements1 = set1.intersection(set2)
        common_elements2 = set1.intersection(set3)
        common_elements3 = set2.intersection(set3)

        # Check if there are any intersections
        if common_elements1 or common_elements2 or common_elements3:
            print("There are intersections:")
            return True
        else:
            print("No intersections found.")
            return False
    
    def train(self, KF_x_train, KF_y_train, KF_x_val, KF_y_val, KF_x_test, KF_y_test, param_dict, result_queue):
        vggish_model = VGGish_Model(num_classes=6, checkpoint_path=vggish_checkpoint_pretrained_model, param_dict=param_dict)
        accuracy, y_pred_list, y_gt_list = vggish_model.train_and_evaluate(x_train=np.array(KF_x_train['Path']), 
                                                                           y_train=np.array([one_hot_encoding_labels(emotion_mapping(x)) for x in KF_y_train]), 
                                                                           validation_data=(np.array(KF_x_val['Path']), np.array([one_hot_encoding_labels(emotion_mapping(x)) for x in KF_y_val])), 
                                                                           x_test=np.array(KF_x_test['Path']),
                                                                           y_test=np.array([one_hot_encoding_labels(emotion_mapping(x)) for x in KF_y_test]),
                                                                           epochs=50,
                                                                           patience=3)
        
        
        result_queue.put([accuracy, y_pred_list, y_gt_list])
    
    def fit(self, dataset):
        for param_id, params in enumerate(product(*self.hyperparameters_grid.values())):
            param_dict = dict(zip(self.hyperparameters_grid.keys(), params))
            print(param_dict)
            # CV among datasets 

            cv_scores_accuracy = []
            y_pred_list = []
            y_gt_list = []
            for train_index, test_index in self.cv.split(dataset, dataset['Emotion']):
                KF_x_train, KF_x_val, KF_y_train, KF_y_val = train_test_split(dataset.iloc[train_index], dataset.iloc[train_index]['Emotion'], stratify=dataset.iloc[train_index]['Emotion'], test_size=0.1, random_state=42)
                
                # augment training data
                KF_x_train = augment_data(KF_x_train, 0.1)
                KF_y_train = KF_x_train['Emotion']

                KF_x_test = dataset.iloc[test_index]
                KF_y_test = dataset.iloc[test_index]['Emotion']

                print(KF_x_train['Emotion'].value_counts())
                print(KF_x_val['Emotion'].value_counts())
                print(KF_x_test['Emotion'].value_counts())

                manager = multiprocessing.Manager()
                result_queue = manager.Queue()
                process = multiprocessing.Process(target=self.train, args=(KF_x_train, KF_y_train, KF_x_val, KF_y_val, KF_x_test, KF_y_test, param_dict, result_queue))
                process.start()
                process.join()

                print("get")
                tmp_queue = result_queue.get()
                accuracy = tmp_queue[0]
                y_pred = tmp_queue[1]
                y_gt = tmp_queue[2]

                print("Accuracy on Test set: ", accuracy)
                cv_scores_accuracy.append(accuracy)

                print(np.array(y_pred).shape, np.array(y_gt).shape)

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

            df1.to_csv(os.path.join(self.path_wav2vec_results, "RAVDESS_hyperparameters_results.csv"), index=False)
            df2.to_pickle(os.path.join(self.path_wav2vec_results, "RAVDESS_pred_gt_lists.pkl"))
        
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


param_grid = {
    'pooling': ["mean", "max"],
    'dense_units': [64, 128], 
    'dropout_rate': [0.3, 0.5],
}

ravSpeech = RAVDESSpeechDataset(args=RAVDESSpeechArgs())
dataset_ravSpeech = prepare_dataframe_for_training(ravSpeech.get_dataset())

ravSong = RAVDESSongDataset(args=RAVDESSongArgs())
dataset_ravSong = prepare_dataframe_for_training(ravSong.get_dataset())

dataset = pd.concat([dataset_ravSpeech, dataset_ravSong], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

gridsearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, model_class=None)
gridsearch.fit(dataset)