import sys, getopt
sys.path.append("../")
import os
import gc
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import product
import importlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.ndimage import zoom
import librosa 
import random
from joblib import Parallel, delayed

seed = 42
torch.manual_seed(seed)

from dataset.RAVDESSpeechDataset import RAVDESSpeechDataset
from dataset.RAVDESSongDataset import RAVDESSongDataset
from config import *
from utils import *
import warnings
warnings.filterwarnings("ignore") 

"""
Comparative Analysis of Traditional,
Hypercomplex, and Pre-trained Deep Neural
Networks for Audio Emotion Recognition

OMAR GHEZZI matr. 984352

MAIN: This script implements the code for the paper 'Comparative Analysis of Traditional, Hypercomplex, and Pre-trained Deep Neural
Networks for Audio Emotion Recognition'. We present a comparative study of traditional, hypercomplex, and pre-trained deep neural networks for the 
Audio Emotion Recognition (AER) task. We introduce novel hypercomplex models, including CliffSER1D, CliffSER2D, PureCliffSER1D and 
PureCliffSER2D, designed to predict discrete emotional classes from hand-crafted MFCCs and logMel spectrogram features in both 1D and 2D domains. 
Our pipeline includes experiments with Empirical Mode Decomposition (EMD) pre-processing analysis. Additionally, we propose 
CliffW2V, a hypercomplex projection-head architecture fine-tuned for AER tasks, built on the end-to-end Wav2Vec pre-trained 
transformer model. Results show that the proposed CliffSER1D, CliffSER2D models outperform traditional approaches, when leveraging 
scalar 1D and 2D hand-crafted logMel features, without EMD preprocessing.
"""

class GridSearchCV:
    def __init__(self, cv, param_grid, module_name, model_name, EMD_flag=False):

        """
        Initialize the GridSearchCV instance.

        Parameters:
        - cv: Cross-validation strategy (e.g., StratifiedKFold)
        - param_grid: Dictionary of hyperparameter grids to search over
        - module_name: Name of the module containing model definitions
        - model_name: Name of the model class within the module
        - EMD_flag: Flag indicating whether to use EMD (Empirical Mode Decomposition) for data multi-channel representation
        """

        self.model_name = model_name

        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        self.trainer_class = getattr(module, 'Trainer')
        self.processor_class = getattr(module, 'Processor')

        self.cv = cv  
        self.param_grid = param_grid
        self.model_class = model_class

        self.results_df = []
        self.df_pred_gt_list = []
        self.best_params = None
        self.best_avg_accuracy = float('-inf')

        self.model_dir = os.path.join('./weights/', self.model_name, '')
        self.path_results = os.path.join('./CV_Clifford_Results', self.model_name, '')
        self.csv_filename = 'Best_hyperparameters_results.csv'
        self.EMD_flag = EMD_flag
        self.visualize = True

    def fit(self, dataset, features_list):

        """
        Perform grid search cross-validation for model training and evaluation.

        Parameters:
        - dataset: Input dataset for training and evaluation
        - features_list: List of features to use from the dataset
        """

        for param_id, params in enumerate(product(*self.param_grid.values())):
            param_dict = dict(zip(self.param_grid.keys(), params))
            print(param_dict)

            cv_scores_accuracy = []
            cv_train_accuracy = []
            y_pred_list = []
            y_gt_list = []
            blades_input_type = convert_blades_to_string(param_dict['blades_idxs'], len(param_dict['g']))
            path_results = os.path.join(self.path_results, 'Blades_'+str(blades_input_type), '')
            for i, (train_index, test_index) in enumerate(self.cv.split(dataset, dataset['Emotion'])):

                print(f"Hyper-params {param_id}: CV fold {i} ...")

                KF_x_train, KF_x_val, _, _ = train_test_split(dataset.iloc[train_index], dataset.iloc[train_index]['Emotion'], stratify=dataset.iloc[train_index]['Emotion'], test_size=0.1, random_state=42)
                KF_x_test = dataset.iloc[test_index]

                if self.EMD_flag:
                    KF_x_train = augment_data_EMD(KF_x_train, 0.1, features_list=features_list)
                else:
                    KF_x_train = augment_data(KF_x_train, 0.1, features_list=features_list)

                print(KF_x_train['Emotion'].value_counts())
                print(KF_x_val['Emotion'].value_counts())
                print(KF_x_test['Emotion'].value_counts())

                # Process EMD-specific data transformations
                if self.EMD_flag:
                    
                    KF_x_train, KF_y_train = from_dataset_to_array_EMD(df=KF_x_train, features_list=features_list)
                    KF_x_val, KF_y_val = from_dataset_to_array_EMD(df=KF_x_val, features_list=features_list)
                    KF_x_test, KF_y_test = from_dataset_to_array_EMD(df=KF_x_test, features_list=features_list)

                    # Handle specific feature types for EMD
                    if 'mel_img_cnn' in features_list['Features'] or 'mfcc_img_cnn' in features_list['Features']:
                        filtered_train = list(filter(lambda x: np.shape(x[0]) == (4, 128, 188), zip(KF_x_train, KF_y_train)))
                        KF_x_train, KF_y_train = zip(*filtered_train)
                        filtered_val = list(filter(lambda x: np.shape(x[0]) == (4, 128, 188), zip(KF_x_val, KF_y_val)))
                        KF_x_val, KF_y_val = zip(*filtered_val)
                        filtered_test = list(filter(lambda x: np.shape(x[0]) == (4, 128, 188), zip(KF_x_test, KF_y_test)))
                        KF_x_test, KF_y_test = zip(*filtered_test)

                        KF_x_train = np.expand_dims(np.array(KF_x_train), axis=3)
                        KF_x_val = np.expand_dims(np.array(KF_x_val), axis=3)
                        KF_x_test = np.expand_dims(np.array(KF_x_test), axis=3)
                        KF_y_train = np.array(KF_y_train)
                        KF_y_val = np.array(KF_y_val)
                        KF_y_test = np.array(KF_y_test)
                    
                    if 'mfcc_vec' in features_list['Features'] or 'mel_vec' in features_list['Features']:
                        filtered_train = list(filter(lambda x: np.shape(x[0]) == (128, 4), zip(KF_x_train, KF_y_train)))
                        KF_x_train, KF_y_train = zip(*filtered_train)
                        filtered_val = list(filter(lambda x: np.shape(x[0]) == (128, 4), zip(KF_x_val, KF_y_val)))
                        KF_x_val, KF_y_val = zip(*filtered_val)
                        filtered_test = list(filter(lambda x: np.shape(x[0]) == (128, 4), zip(KF_x_test, KF_y_test)))
                        KF_x_test, KF_y_test = zip(*filtered_test)

                        KF_x_train, KF_x_val, KF_x_test = np.array(KF_x_train), np.array(KF_x_val), np.array(KF_x_test)
                        KF_y_train, KF_y_val, KF_y_test = np.array(KF_y_train), np.array(KF_y_val), np.array(KF_y_test)
                    
                    mean_train = np.mean(KF_x_train, axis=0, keepdims=True)
                    std_train = np.std(KF_x_train, axis=0, keepdims=True)
                
                # data derivatives transformations
                elif 'delta1_mel_img_cnn' in features_list['Features']:

                    new_features_list = [{'Features': [feature]} for feature in features_list['Features']]

                    x_train, x_val, x_test = [], [], []
                    y_train, y_val, y_test = [], [], []

                    for feature in new_features_list:
                        for df, x, y in [(KF_x_train, x_train, y_train), (KF_x_val, x_val, y_val), (KF_x_test, x_test, y_test)]:
                            x_data, y_data = from_dataset_to_array(df=df, features_list=feature)
                            x.append(x_data)
                            y.append(y_data)
                    
                    KF_x_train, KF_y_train = np.array(x_train), np.array(y_train)[0,:,:].squeeze()
                    KF_x_val, KF_y_val = np.array(x_val), np.array(y_val)[0,:,:].squeeze()
                    KF_x_test, KF_y_test = np.array(x_test), np.array(y_test)[0,:,:].squeeze()

                    KF_x_train = KF_x_train.reshape([KF_x_train.shape[1], KF_x_train.shape[0], KF_x_train.shape[2], KF_x_train.shape[3]])
                    KF_x_val = KF_x_val.reshape([KF_x_val.shape[1], KF_x_val.shape[0], KF_x_val.shape[2], KF_x_val.shape[3]])
                    KF_x_test = KF_x_test.reshape([KF_x_test.shape[1], KF_x_test.shape[0], KF_x_test.shape[2], KF_x_test.shape[3]])

                    mean_train = np.mean(KF_x_train, axis=0, keepdims=True)
                    std_train = np.std(KF_x_train, axis=0, keepdims=True)

                # Regular data transformations
                else:
                    
                    KF_x_train, KF_y_train = from_dataset_to_array(df=KF_x_train, features_list=features_list)
                    KF_x_val, KF_y_val = from_dataset_to_array(df=KF_x_val, features_list=features_list)
                    KF_x_test, KF_y_test = from_dataset_to_array(df=KF_x_test, features_list=features_list)

                    KF_x_train = np.expand_dims(KF_x_train, axis=2)
                    KF_x_val = np.expand_dims(KF_x_val, axis=2)
                    KF_x_test = np.expand_dims(KF_x_test, axis=2) 

                    mean_train = np.mean(KF_x_train, axis=0, keepdims=True)
                    std_train = np.std(KF_x_train, axis=0, keepdims=True)

                    # X = KF_x_train.squeeze() 
                    # y = KF_y_train.squeeze()
                    # conf_matrix, acc = cluster_images(X, y)

                    #import code; code.interact(local=locals())

                #train_sum = np.sum(KF_x_train, axis=0, keepdims=True)
                #train_sum[train_sum == 0] = 1e-10

                # import code; code.interact(local=locals()) 
                # import seaborn as sns
                # sns.set_context('poster')
                # plt.figure()
                # librosa.display.specshow(KF_x_train[0,0,:,:], sr=8000, x_axis='time', y_axis='mel')
                # librosa.display.specshow(KF_x_train[1000,3,:,:,:].squeeze(), sr=8000, x_axis='time', y_axis='mel')
                # plt.title('mel power spectrogram')
                # plt.imshow(KF_x_train[0,0,:,:], aspect='auto', cmap='viridis')
                # plt.colorbar(format='%+02.0f dB')
                # plt.tight_layout()
                # plt.savefig('MFFCC_1DCNN_EMD_0.pdf')

                # Dataset standardization

                KF_x_train = (KF_x_train - mean_train) / std_train
                KF_x_val = (KF_x_val - mean_train) / std_train
                KF_x_test = (KF_x_test - mean_train) / std_train

                print('Train: {}, {}, Val: {}, {}, Test: {}, {}.'.format(KF_x_train.shape, KF_y_train.shape, KF_x_val.shape, KF_y_val.shape, KF_x_test.shape, KF_y_test.shape))

                batch_size = 16
                weight_decay = 0.0005
                max_epochs = 500
                lr = 0.001
                loss_type = 'CrossEntropy'

                train_dataset = TensorDataset(torch.FloatTensor(KF_x_train), torch.FloatTensor(KF_y_train))
                val_dataset = TensorDataset(torch.FloatTensor(KF_x_val), torch.FloatTensor(KF_y_val))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                path = os.path.join(self.model_dir, convert_signature_to_string(param_dict['g']), convert_blades_to_string(param_dict['blades_idxs'], 
                                len(param_dict['g'])), 'hyper_n'+str(param_id), 'fold_n'+str(i), '')
                trainer = self.trainer_class(self.model_class, param_dict, loss_type, max_epochs=max_epochs, batch_size=batch_size, model_dir=path, visualize=self.visualize)
                train_loss, train_acc = trainer.train(train_loader, val_loader, lr=lr, weight_decay=weight_decay)

                test_dataset = TensorDataset(torch.FloatTensor(KF_x_test), torch.FloatTensor(KF_y_test))
                test_loader = DataLoader(test_dataset, batch_size=1)

                tester = self.processor_class(self.model_class, param_dict, loss_type, load_path=path, type='epoch', use_last_epoch=False)
                y_pred = tester.predict(test_loader)
                if self.EMD_flag:
                    y_pred = y_pred.reshape(KF_y_test.shape)
                
                test_loss, accuracy = tester.score(y_pred, KF_y_test)
                
                #y_pred_classes = self.probabilities_to_one_hot(y_pred)
                #accuracy = self.compute_accuracy(y_pred_classes, KF_y_test)

                print("Accuracy on Test set: ", accuracy)
                cv_scores_accuracy.append(accuracy)
                cv_train_accuracy.append(train_acc)

                #y_pred_list.append(y_pred_classes)
                y_pred_list.append(y_pred)
                y_gt_list.append(KF_y_test)
                
                del KF_x_train, KF_x_val, KF_y_train, KF_y_val #, rlrp, early_stopping
                gc.collect()

            # Calculate and print the mean and standard deviation of the cross-validation scores
            mean_train_accuracy = np.mean(cv_train_accuracy)
            std_train_accuracy = np.std(cv_scores_accuracy)
            mean_accuracy = np.mean(cv_scores_accuracy)
            std_accuracy = np.std(cv_scores_accuracy)
            print(f'Mean Accuracy: {mean_accuracy:.4f}')
            print(f'Accuracy Standard Deviation: {std_accuracy:.4f}')
            
            avg_metrics = calculate_avg_metrics(y_pred_list, y_gt_list)
            param_dict['avg_train_accuracy'] = mean_train_accuracy
            param_dict['std_train_accuracy'] = std_train_accuracy
            param_dict['avg_test_accuracy'] = mean_accuracy
            param_dict['std_test_accuracy'] = std_accuracy
            param_dict['avg_test_accuracy_post'] = avg_metrics['avg_accuracy']
            param_dict['avg_precision_post'] = avg_metrics['avg_precision']
            param_dict['avg_recall_post'] = avg_metrics['avg_recall']
            param_dict['avg_f1_post'] = avg_metrics['avg_f1']

            if self.best_avg_accuracy < mean_accuracy:
                self.best_avg_accuracy = mean_accuracy
                self.best_params = param_dict

             # Append the new experiment to the DataFrame
            self.results_df.append(param_dict)
            self.df_pred_gt_list.append({'param_id': param_id, 'y_pred_list':y_pred_list, 'y_gt_list':y_gt_list})

            df1 = pd.DataFrame(self.results_df)
            df2 = pd.DataFrame(self.df_pred_gt_list)

            # Save the updated DataFrame to a CSV file
            # Check if directory exists
            if not os.path.exists(path_results):
                # If directory does not exist, create it
                os.makedirs(path_results)

            df1.to_csv(os.path.join(path_results, "RAVDESS_hyperparameters_results.csv"), index=False)
            df2.to_pickle(os.path.join(path_results, "RAVDESS_pred_gt_lists.pkl"))
        
        # Check if the file exists, and create it if it doesn't
        file_exists = os.path.isfile(os.path.join(self.path_results, self.csv_filename))
        # Open the CSV file in append mode and write the row
        with open(os.path.join(self.path_results, self.csv_filename), 'w', newline='') as csvfile:
            fieldnames = self.best_params.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # If the file doesn't exist, write the header row first
            if not file_exists:
                writer.writeheader() 
            # Write the new row to the CSV file
            writer.writerow(self.best_params)


def main(argv):

    what = 0 # Default action

    # Parse command line arguments
    opts, args = getopt.getopt(argv,"ha:e:", ["action=", "emd="])
    for opt, arg in opts:
        if opt == '-h':
            print ('run_all.py -a <action> -emd <EMD flag (bool)>')
            sys.exit()
        elif opt in ("-a", "--action"):
            try:
                what = int(arg)
            except ValueError:
                print('The action parameter must be an integer.')
                sys.exit(2)
        elif opt in ("-e", "--emd"):
            if arg.lower() in ('true', 't', 'y', 'yes'):
                EMD_flag = True
            elif arg.lower() in ('false', 'f', 'n', 'no'):
                EMD_flag = False
            else:
                print('The EMD flag must be a boolean (true/false, 1/0).')
                sys.exit(2)
    print ('\nAction is ', what)
    print ('\nEMD is', EMD_flag)

    # Initialize a list of datasets
    ravSpeech = RAVDESSpeechDataset(args=RAVDESSpeechArgs(), EMD=EMD_flag, features_list=['mfcc_vec', 'delta1_mfcc_vec', 'delta2_mfcc_vec', 'stft_vec', 'chroma_vec', 'mel_vec', 
                                    'contrast_vec', 'mfcc_img_cnn', 'mel_img_cnn'])
    dataset_ravSpeech = prepare_dataframe_for_training(ravSpeech.get_dataset())

    ravSong = RAVDESSongDataset(args=RAVDESSongArgs(), EMD=EMD_flag, features_list=['mfcc_vec', 'delta1_mfcc_vec', 'delta2_mfcc_vec', 'stft_vec', 'chroma_vec', 'mel_vec', 
                                    'contrast_vec', 'mfcc_img_cnn', 'mel_img_cnn'])
    dataset_ravSong = prepare_dataframe_for_training(ravSong.get_dataset()) 

    # Combine datasets
    dataset = pd.concat([dataset_ravSpeech, dataset_ravSong], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if what == 0:

        # Initialize one module name
        model_name = 'CliffSER1D_EMD'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [8, 15], 
            'stride': [1, 2],
            'dense_units': [[32, 64, 128, 256, 128, 64]], #[[16, 32, 64, 128, 128, 256], [8, 16, 32, 64, 64, 128], [2, 4, 8, 16, 16, 64], [1, 1, 1, 1, 1, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0, 1, 2, 3]],
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mfcc_vec']} #'mel_vec', 'mfcc_vec'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    if what == 1:

        # Initialize one module name
        model_name = 'PureCliffSER1D_EMD'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [8, 15], 
            'stride': [1, 2],
            'dense_units': [[1, 1, 1, 1, 256, 128, 64]], #[[16, 32, 64, 128, 128, 256], [8, 16, 32, 64, 64, 128], [2, 4, 8, 16, 16, 64], [1, 1, 1, 1, 1, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0, 1, 2, 3]],
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mfcc_vec']} #'mel_vec', 'mfcc_vec'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    if what == 2:

        # Initialize one module name
        model_name = 'CliffSER1D'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define the hyperparameters grids

        #       x       =   x_0 + x1e1 + x2e2 + x12e12
        # 'blades_idxs' :   0     1      2      3 

        #       x       =   x_0 + x1e1 + x2e2 + x3e3 + x12e12 + x13e13 + x23e23 + x123e123
        # 'blades_idxs' :   0     1      2      3      4        5        6        7

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [8, 15], 
            'stride': [1, 2],
            'dense_units': [[32, 64, 128, 256, 128, 64]], #[[16, 32, 64, 128, 128, 256], [8, 16, 32, 64, 64, 128], [2, 4, 8, 16, 16, 64], [1, 1, 1, 1, 1, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0]],
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mel_vec']} #'mel_vec', 'mfcc_vec'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    if what == 3:

        # Initialize one module name
        model_name = 'PureCliffSER1D'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [8, 15], 
            'stride': [1, 2],
            'dense_units': [[1, 1, 1, 1, 256, 128, 64]], #[[16, 32, 64, 128, 128, 256], [8, 16, 32, 64, 64, 128], [2, 4, 8, 16, 16, 64], [1, 1, 1, 1, 1, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0]],
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mel_vec']} #'mel_vec', 'mfcc_vec'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    
    if what == 4:

        model_name = 'CliffSER2D_EMD'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define the hyperparameters grids

        #       x       =   x_0 + x1e1 + x2e2 + x12e12
        # 'blades_idxs' :   0     1      2      3 

        #       x       =   x_0 + x1e1 + x2e2 + x3e3 + x12e12 + x13e13 + x23e23 + x123e123
        # 'blades_idxs' :   0     1      2      3      4        5        6        7

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [3, 5],
            'stride': [2],
            'dense_units': [[32, 64, 128, 256, 128, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0, 1, 2, 3]], #VLF, HF, MF, LF
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mel_img_cnn']}

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    if what == 5:

        model_name = 'CliffSER2D'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define the hyperparameters grids

        #       x       =   x_0 + x1e1 + x2e2 + x12e12
        # 'blades_idxs' :   0     1      2      3 

        #       x       =   x_0 + x1e1 + x2e2 + x3e3 + x12e12 + x13e13 + x23e23 + x123e123
        # 'blades_idxs' :   0     1      2      3      4        5        6        7

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [3, 5],
            'stride': [2],
            'dense_units': [[32, 64, 128, 256, 128, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0]], #VLF, HF, MF, LF
            'sandwich': [False] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mfcc_img_cnn']} #'mfcc_img_cnn', 'mel_img_cnn'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)
    
    if what == 6:

        model_name = 'PureCliffSER2D'
        
        # Specify which module contains the model
        module_name = 'clifford_'

        # Define the hyperparameters grids

        #       x       =   x_0 + x1e1 + x2e2 + x12e12
        # 'blades_idxs' :   0     1      2      3 

        #       x       =   x_0 + x1e1 + x2e2 + x3e3 + x12e12 + x13e13 + x23e23 + x123e123
        # 'blades_idxs' :   0     1      2      3      4        5        6        7

        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'kernel_size': [3, 5],
            'stride': [1, 2],
            'dense_units': [[1, 1, 1, 1, 256, 128, 64]],
            'g': [[1,1], [1, 1, 1], [-1,-1], [-1, -1, -1]], 
            'blades_idxs': [[0]], 
            'sandwich': [True] #[True, False]
        } 

        # Initialize a list of features
        features_list =  {'Features': ['mel_img_cnn']} #'mfcc_img_cnn', 'mel_img_cnn'

		# Perform training using a Grid search cross-validation
        gridSearch = GridSearchCV(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), param_grid=param_grid, module_name=module_name, model_name=model_name, EMD_flag=EMD_flag)
        gridSearch.fit(dataset, features_list)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main(sys.argv[1:])
