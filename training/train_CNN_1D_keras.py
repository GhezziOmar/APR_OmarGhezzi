import sys
sys.path.append("../")

import numpy as np
import pandas as pd

# local imports
from dataset.RAVDESSpeechDataset import RAVDESSpeechDataset
from dataset.RAVDESSongDataset import RAVDESSongDataset
from config import *
from utils import *

from sklearn.model_selection import StratifiedKFold
from models.Model_CNN_1D import CNN1D_Model
from itertools import product
import os
from utils import *
import tensorflow as tf
import gc
import keras as K
import csv

class GridSearchCV:
    def __init__(self, cv, param_grid, model_class, model_name):
        self.model_name = model_name

        self.cv = cv  
        self.param_grid = param_grid
        self.model_class = model_class

        self.results_df = []
        self.df_pred_gt_list = []
        self.best_params = None
        self.best_avg_accuracy = float('-inf')

        self.path_cnn2d_results = os.path.join('../../CV_Results', self.model_name)
        self.csv_filename = 'Best_hyperparameters_results.csv'

    def probabilities_to_one_hot(self, probabilities):
        # Convert each row in the 2D array to one-hot encoding
        one_hot_encoded = np.zeros_like(probabilities)
        one_hot_encoded[np.arange(len(probabilities)), probabilities.argmax(axis=1)] = 1

        return one_hot_encoded

    def compute_accuracy(self, predictions, ground_truths):
        # Ensure that the lengths of both lists are the same
        if len(predictions) != len(ground_truths):
            raise ValueError("Lengths of predictions and ground_truths must be the same.")

        # Initialize a variable to count the correct predictions
        correct_predictions = 0

        # Iterate through each pair of prediction and ground truth vectors
        for pred, gt in zip(predictions, ground_truths):
            # Check if the one-hot encoding is the same for both prediction and ground truth
            #print(pred, gt)
            if np.argmax(pred) == np.argmax(gt):
                correct_predictions += 1

        # Calculate the accuracy as the percentage of correct predictions
        accuracy = (correct_predictions / len(predictions)) * 100.0

        return accuracy

    def fit(self, dataset, features_list):
        for param_id, params in enumerate(product(*self.param_grid.values())):
            param_dict = dict(zip(self.param_grid.keys(), params))
            print(param_dict)

            cv_scores_accuracy=[]
            y_pred_list = []
            y_gt_list = []
            for train_index, test_index in self.cv.split(dataset, dataset['Emotion']):
                KF_x_train, KF_x_val, _, _ = train_test_split(dataset.iloc[train_index], dataset.iloc[train_index]['Emotion'], stratify=dataset.iloc[train_index]['Emotion'], test_size=0.1, random_state=42)
                KF_x_test = dataset.iloc[test_index]

                KF_x_train = augment_data(KF_x_train, 0.1, features_list=features_list)

                print(KF_x_train['Emotion'].value_counts())
                print(KF_x_val['Emotion'].value_counts())
                print(KF_x_test['Emotion'].value_counts())

                KF_x_train, KF_y_train = from_dataset_to_array(df=KF_x_train, features_list=features_list)
                KF_x_val, KF_y_val = from_dataset_to_array(df=KF_x_val, features_list=features_list)
                KF_x_test, KF_y_test = from_dataset_to_array(df=KF_x_test, features_list=features_list)

                KF_x_train = np.expand_dims(KF_x_train, axis=2)
                KF_x_val = np.expand_dims(KF_x_val, axis=2)
                KF_x_test = np.expand_dims(KF_x_test, axis=2)

                print(KF_x_train.shape, KF_x_val.shape, KF_y_train.shape, KF_y_val.shape, KF_x_test.shape, KF_y_test.shape)

                model = self.model_class()
                model.set_hyperparameters(param_dict)

                rlrp = K.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.4, verbose=0, patience=2, min_lr=0.000001)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
                model.fit(x_train=KF_x_train, y_train=KF_y_train, batch_size=20, validation_data=(KF_x_val, KF_y_val), epochs=500, validation_batch_size=20, callbacks=[rlrp, early_stopping])

                y_pred = model.predict(KF_x_test)
                y_pred_classes = self.probabilities_to_one_hot(y_pred)
                accuracy = self.compute_accuracy(y_pred_classes, KF_y_test)

                print("Accuracy on Test set: ", accuracy)
                cv_scores_accuracy.append(accuracy)

                y_pred_list.append(y_pred_classes)
                y_gt_list.append(KF_y_test)
                
                del model, KF_x_train, KF_x_val, KF_y_train, KF_y_val, rlrp, early_stopping
                gc.collect()

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
            if not os.path.exists(self.path_cnn2d_results):
                # If directory does not exist, create it
                os.makedirs(self.path_cnn2d_results)

            df1.to_csv(os.path.join(self.path_cnn2d_results, "RAVDESS_hyperparameters_results.csv"), index=False)
            df2.to_pickle(os.path.join(self.path_cnn2d_results, "RAVDESS_pred_gt_lists.pkl"))
        
        # Check if the file exists, and create it if it doesn't
        file_exists = os.path.isfile(os.path.join(self.path_cnn2d_results, self.csv_filename))
        # Open the CSV file in append mode and write the row
        with open(os.path.join(self.path_cnn2d_results, self.csv_filename), 'a', newline='') as csvfile:
            fieldnames = self.best_params.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file doesn't exist, write the header row first
            if not file_exists:
                writer.writeheader()

            # Write the new row to the CSV file
            writer.writerow(self.best_params)

param_grid = {
    'kernel_size': [10, 8], 
    'conv_stride_kernel': [1, 2],
    'pool_size': [2, 3],
    'pool_stride': [1, 2],
    'dense_units': [(128,64), (256,128)]
}

features_list =  {'Features': ['mfcc_vec']}
ravSpeech = RAVDESSpeechDataset(args=RAVDESSpeechArgs(), features_list=['mfcc_vec', 'delta1_mfcc_vec', 'delta2_mfcc_vec', 'stft_vec', 'chroma_vec', 'mel_vec', 'contrast_vec', 'mfcc_img_cnn', 'mel_img_cnn'])
dataset_ravSpeech = prepare_dataframe_for_training(ravSpeech.get_dataset())

ravSong = RAVDESSongDataset(args=RAVDESSongArgs(), features_list=['mfcc_vec', 'delta1_mfcc_vec', 'delta2_mfcc_vec', 'stft_vec', 'chroma_vec', 'mel_vec', 'contrast_vec', 'mfcc_img_cnn', 'mel_img_cnn'])
dataset_ravSong = prepare_dataframe_for_training(ravSong.get_dataset())

dataset = pd.concat([dataset_ravSpeech, dataset_ravSong], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, model_class=CNN1D_Model, model_name="CNN1D")
gridSearch.fit(dataset, features_list)