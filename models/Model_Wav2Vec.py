from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

import torchaudio
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from datasets import load_dataset
import os

import torch
from config import *
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor

from config import *

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

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
    
class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

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
    
class DataPreparation():
    def __init__(self, dataframe_orig, input_column='path', output_column='emotion'):
        self.processor = None
        self.config = None
        
        self.label_list = None
        self.num_labels = None
        self.target_sampling_rate = None

        self.train_dataset = None
        self.eval_dataset = None
        
        self.dataframe_orig = dataframe_orig
        self.input_column = input_column
        self.output_column = output_column

        if (os.path.exists(f"{save_path_csv}/train.csv") and os.path.exists(f"{save_path_csv}/test.csv")) == False:
            self.generate_csv(dataframe_orig)
    
        self.train_dataset, self.eval_dataset, self.processor, self.config, self.label_list, self.num_labels, self.target_sampling_rate = self.prepare_dataset_for_training()
    
    def generate_csv(self, df):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["emotion"])

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(f"{save_path_csv}/train.csv", sep="\t", encoding="utf-8", index=False)
        test_df.to_csv(f"{save_path_csv}/test.csv", sep="\t", encoding="utf-8", index=False)

        #print(train_df.shape)
        #print(test_df.shape)
    
    def prepare_dataset_for_training(self):
        def speech_file_to_array_fn(path):
            speech_array, sampling_rate = torchaudio.load(path)
            print(path)
            if len(speech_array.size()) > 1:
                speech_array = torch.mean(speech_array, axis=0)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            return speech

        def label_to_id(label, label_list):
            if len(label_list) > 0:
                return label_list.index(label) if label in label_list else -1

            return label

        def preprocess_function(examples):
            speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
            target_list = [label_to_id(label, label_list) for label in examples[output_column]]

            result = processor(speech_list, sampling_rate=target_sampling_rate)
            result["labels"] = list(target_list)

            return result
    
        data_files = {
            "train": f"{save_path_csv}/train.csv",
            "validation": f"{save_path_csv}/test.csv",
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        train = dataset["train"]
        eval = dataset["validation"]

        # we need to distinguish the unique labels in our SER dataset
        label_list = train.unique(output_column)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")
        
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        target_sampling_rate = processor.feature_extractor.sampling_rate
        print(f"The target sampling rate: {target_sampling_rate}")

        train_dataset = train.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        eval_dataset = eval.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        return train_dataset, eval_dataset, processor, label_list, num_labels, target_sampling_rate

    def get_dataset_info(self):
        return self.label_list, self.num_labels, self.target_sampling_rate
    
    def get_datasets_train_eval(self):
        return self.train_dataset, self.eval_dataset
    
    def get_processor_config(self):
        return self.processor, self.config