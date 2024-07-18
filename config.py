from dataclasses import dataclass
import os

seed = 42
selected_emotion = {'Anger': "01",
                    'Disgust': "02",
                    'Fear': "03",
                    'Happy': "04",
                    'Neutral': "05",
                    'Sad': "06"
                    }

path_to_dir = "./" # absolute path to the project directory

vggish_checkpoint_pretrained_model = os.path.join(path_to_dir, "project/models/vggish_utils/vggish_model.ckpt")

# WAV2VEC
save_path_csv = os.path.join(path_to_dir, "./wav2vec_utils")
model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
output_dir= os.path.join(path_to_dir, "./wav2vec_utils/saved_models")
input_column = "Path"
output_column = "Emotion"

@dataclass
class Args:
   pass

class RAVDESSpeechArgs(Args):
   data_path: str = os.path.join(path_to_dir, 'RAVDESS_Speech')  # path to dataset
   dataset_name: str = 'RAVDESSpeech'  # dataset name
   save_path_dataset: str = "./dataset/Datasets_pkl/"  # path to save dataset
   save_path_dataset_EMD : str = "./dataset/Datasets_pkl_EMD/" # path to save the EMD dataset

class RAVDESSongArgs(Args):
   data_path: str = os.path.join(path_to_dir, 'RAVDESS_Song')  # path to dataset
   dataset_name: str = 'RAVDESSong'  # dataset name
   save_path_dataset: str = "./dataset/Datasets_pkl/"  # path to save dataset
   save_path_dataset_EMD: str = "./dataset/Datasets_pkl_EMD/"  # path to save the EMD dataset