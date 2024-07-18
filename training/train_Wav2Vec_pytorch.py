import sys
sys.path.append("../")

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

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.dense_1_out)
        self.dropout = nn.Dropout(config.dropout_1)
        self.out_proj = nn.Linear(config.dense_1_out, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def summary_classifier(self):
        summary(self.classifier, input_size=(self.config.hidden_size,))

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch
    
class MyCallbacks(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.cv_scores_accuracy = 0
        
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        self.cv_scores_accuracy = metrics['test_accuracy']

    def get_cv_scores_accuracy(self):
        return self.cv_scores_accuracy

class Wav2Vec_Model():
    def __init__(self, hyperparameters=None, mycallbacks=None):
        self.mycallbacks = mycallbacks

        self.y_pred = None
        self.y_gt = None

        self.num_labels = 6
        self.label_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
       
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            label2id={label: i for i, label in enumerate(self.label_list)},
            id2label={i: label for i, label in enumerate(self.label_list)},
        )

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.0, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
        ])

        # Hyperparameters 
        if hyperparameters is not None:
            setattr(self.config, 'pooling_mode', hyperparameters['pooling_mode'])
            setattr(self.config, 'dense_1_out',  hyperparameters['dense_1_out'])
            setattr(self.config, 'dropout_1', hyperparameters['dropout_1'])
        else:
            setattr(self.config, 'pooling_mode', 'mean')
            setattr(self.config, 'dense_1_out', 200)
            setattr(self.config, 'dropout_1', 0.3)

    def set_hyperparameters(self, hyperparameters):
        if 'pooling_mode' in hyperparameters:
            setattr(self.config, 'pooling_mode', hyperparameters['pooling_mode'])
        if 'dense_1_out' in hyperparameters:
            setattr(self.config, 'dense_1_out',  hyperparameters['dense_1_out'])
        if 'dropout_1' in hyperparameters:
            setattr(self.config, 'dropout_1', hyperparameters['dropout_1'])

    def prepare_dataset_for_training(self, dataset_wav2Vec_train, dataset_wav2Vec_eval, dataset_wav2Vec_test, augmentation):
        def speech_file_to_array_fn(path, to_augment=None):
            if augmentation:
                speech_array, sampling_rate = torchaudio.load(path)
                if speech_array.size()[0] > 1:
                    speech_array = torch.mean(speech_array, axis=0)
                if to_augment:
                    speech_array = torch.from_numpy(self.augment(samples=np.array(speech_array), sample_rate=sampling_rate))
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech
            else:
                speech_array, sampling_rate = torchaudio.load(path)
                if speech_array.size()[0] > 1:
                    speech_array = torch.mean(speech_array, axis=0)
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech


        def label_to_id(label, label_list):
            if len(label_list) > 0:
                return label_list.index(label) if label in label_list else -1

            return label

        def preprocess_function(examples):
            if augmentation:
                speech_list = [speech_file_to_array_fn(path, to_augment) for path, to_augment in zip(examples["Path"], examples["Augmented"])]
            else:
                speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
            target_list = [label_to_id(label, self.label_list) for label in examples[output_column]]

            result = processor(speech_list, sampling_rate=target_sampling_rate)
            result["labels"] = list(target_list)

            return result

        def augmentation_wav2vec(X_df, percentage_augmentation, dataset_name):
            df_augmented_features = []
            X_df = X_df.assign(Augmented=False)
            if dataset_name == "train":
                for emotion in X_df['Emotion'].unique():
                    df_emotion = X_df[X_df['Emotion'] == emotion]
                    number_of_augmented_samples_emotion_class = int((len(X_df) * percentage_augmentation)/len(X_df['Emotion'].unique()))
                    for _ in range(number_of_augmented_samples_emotion_class+abs(np.max(X_df['Emotion'].value_counts())-len(df_emotion))):
                        random_index = np.random.randint(0, len(df_emotion))
                        dict_features = df_emotion.iloc[random_index].to_dict()
                        dict_features['Augmented'] = True
                        df_augmented_features.append(dict_features)
                
                return pd.concat([X_df, pd.DataFrame(df_augmented_features)]).sample(frac=1, random_state=42).reset_index(drop=True)

            return X_df

    
        if (os.path.exists(f"{save_path_csv}/train.csv") and os.path.exists(f"{save_path_csv}/eval.csv") and os.path.exists(f"{save_path_csv}/test.csv")) == False:
            if augmentation:
                dataset_wav2Vec_train = augmentation_wav2vec(dataset_wav2Vec_train, 0.1, "train") # just here
                dataset_wav2Vec_eval = augmentation_wav2vec(dataset_wav2Vec_eval, 0.1, "eval") # no here
                dataset_wav2Vec_test = augmentation_wav2vec(dataset_wav2Vec_test, 0.1, "test") # no here
            # train eval 
            train_df = dataset_wav2Vec_train.reset_index(drop=True)
            eval_df = dataset_wav2Vec_eval.reset_index(drop=True)
            train_df.to_csv(f"{save_path_csv}/train.csv", sep="\t", encoding="utf-8", index=False)
            eval_df.to_csv(f"{save_path_csv}/eval.csv", sep="\t", encoding="utf-8", index=False)
            # test
            dataset_wav2Vec_test = dataset_wav2Vec_test.reset_index(drop=True)
            dataset_wav2Vec_test.to_csv(f"{save_path_csv}/test.csv", sep="\t", encoding="utf-8", index=False)


        data_files = {
            "train": f"{save_path_csv}/train.csv",
            "validation": f"{save_path_csv}/eval.csv",
            "test": f"{save_path_csv}/test.csv"
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        train = dataset["train"]
        eval = dataset["validation"]
        test = dataset["test"]

        print("Train dataset: ", len(train))
        print("Eval dataset: ", len(eval))
        print("Test dataset: ", len(test))

        print(pd.Series(train['Emotion']).value_counts())
        print(pd.Series(eval['Emotion']).value_counts())
        print(pd.Series(test['Emotion']).value_counts())
        
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        target_sampling_rate = processor.feature_extractor.sampling_rate
        print(f"The target sampling rate: {target_sampling_rate}")

        test_dataset = test.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=10
        )

        train_dataset = train.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=10
        )

        eval_dataset = eval.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=10
        )

        # remove csv files
        for file_path in [os.path.join(save_path_csv, "train.csv"), os.path.join(save_path_csv, "eval.csv"), os.path.join(save_path_csv, "test.csv")]:
            try:
                # Remove (delete) the file
                os.remove(file_path)
                print(f"The file {file_path} has been removed.")
            except FileNotFoundError:
                print(f"The file {file_path} does not exist.")
        
        return train_dataset, eval_dataset, test_dataset, processor
    
    def fit(self, dataset_wav2Vec_train, dataset_wav2Vec_eval, dataset_wav2Vec_test):
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)

            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # get data
        train_dataset, eval_dataset, test_dataset, processor = self.prepare_dataset_for_training(dataset_wav2Vec_train, dataset_wav2Vec_eval, dataset_wav2Vec_test, augmentation=True)
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        # get model
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        # freeze feature extractor
        model.freeze_feature_extractor()

        # summary model
        model.summary_classifier()

        # prepare training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=30,
            fp16=True,
            logging_strategy="epoch",
            learning_rate=1e-4,
            save_total_limit=1,
            metric_for_best_model="eval_accuracy",
            load_best_model_at_end=True,
        )

        # prepare Trainer
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), self.mycallbacks]
        )

        # train model
        trainer.train()

        # test model
        test_results = trainer.predict(test_dataset)

        self.y_pred = np.argmax(test_results.predictions, axis=1)
        self.y_gt = test_results.label_ids

    def get_y_pred(self):
        return self.y_pred
    
    def get_y_gt(self):
        return self.y_gt

class GridSearchCV_Wav2Vec():
    def __init__(self, cv, hyperparameters_grid, datasets):
        self.hyperparameters_grid = hyperparameters_grid
        self.datasets = datasets
        self.cv = cv

        self.results_df = []
        self.df_pred_gt_list = []
        self.best_avg_accuracy = float('-inf')

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
        model = Wav2Vec_Model(param_dict, mycallbacks)
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

                mycallbacks = MyCallbacks()
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

def main():
    param_grid = {
        'pooling_mode': ["mean", "max"],
        'dense_1_out': [100, 200], 
        'dropout_1': [0.3, 0.5],
    }

    datasets = 'RAVDESS'
    grid_search_wav2vec = GridSearchCV_Wav2Vec(cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), hyperparameters_grid=param_grid, datasets=datasets)
    grid_search_wav2vec.fit()

process = multiprocessing.Process(target=main)
process.start()