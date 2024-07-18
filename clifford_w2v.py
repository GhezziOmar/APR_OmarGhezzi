import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import math
from torchvision import transforms
import torchaudio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn import init
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Union
import shutil

seed = 42
torch.manual_seed(seed)

import sys
sys.path.append("../")

from transformers.trainer_callback import TrainerControl, TrainerState
from models.Model_Wav2Vec import DataCollatorCTCWithPadding
from models.Model_Wav2Vec import Wav2Vec2ForSpeechClassification
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2Processor, AutoConfig, TrainerCallback
from datasets import load_dataset

from config import *
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from utils import emotion_mapping_wav2vec

from dataset.RAVDESSpeechDataset import RAVDESSpeechDataset
from dataset.RAVDESSongDataset import RAVDESSongDataset

from utils import calculate_avg_metrics
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from torchinfo import summary


from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        return

    def forward(self, preds, labels):
        return F.cross_entropy(preds.softmax(dim=1), labels)

# ########################################Clifford things##################################################
    
def _w_assert(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Convert Clifford weights to tensor .
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Clifford weights.
    Raises:
        ValueError: Unknown weight type.
    Returns:
        torch.Tensor: Clifford weights as torch.Tensor.
    """
    if type(w) in (tuple, list):
        w = torch.stack(w)
        return w
    elif isinstance(w, torch.Tensor):
        return w
    elif isinstance(w, nn.Parameter):
        return w
    elif isinstance(w, nn.ParameterList):
        return w
    else:
        raise ValueError("Unknown weight type.")

class CliffordSignature:
    def __init__(self, g: Union[tuple, list, torch.Tensor]):
        super().__init__()
        self.g = self._g_tensor(g)
        self.dim = self.g.numel()
        if self.dim == 1:
            self.n_blades = 2
        elif self.dim == 2:
            self.n_blades = 4
        elif self.dim == 3:
            self.n_blades = 8
        else:
            raise NotImplementedError("Wrong Clifford signature.")

    def _g_tensor(self, g: Union[tuple, list, torch.Tensor]) -> torch.Tensor:
        """Convert Clifford signature to tensor.
        Args:
            g (Union[tuple, list, torch.Tensor]): Clifford signature.
        Raises:
            ValueError: Unknown metric.
        Returns:
            torch.Tensor: Clifford signature as torch.Tensor.
        """
        if type(g) in (tuple, list):
            g = torch.as_tensor(g, dtype=torch.float32)
        elif isinstance(g, torch.Tensor):
            pass
        else:
            raise ValueError("Unknown signature.")
        if not torch.any(abs(g) == 1.0):
            raise ValueError("Clifford signature should have at least one element as 1.")
        return g

def get_2d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebras, g = [-1, -1] corresponds to a quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3], g[1] * w[2]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0], -g[0] * w[1]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k

def get_2d_clifford_rotation_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Rotational Clifford kernel for 2d Clifford algebras, the vector part corresponds to quaternion rotation.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(6, d~input~, d~output~, ...)`.
                    `w[0]`, `w[1]`, `w[2]`, `w[3]` are the 2D Clifford weight tensors;
                    `w[4]` is the scaling tensor; `w[5]` is the zero kernel tensor.

        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    assert g[0] == -1 and g[1] == -1, "Wrong signature of Clifford algebra. Signature not suitable for rotation kernel."
    w = _w_assert(w)
    assert len(w) == 6

    # Adding scalar output kernel.
    k0 = torch.cat([w[0], -w[1], -w[2], -w[3]], dim=1)

    # Rotational kernel from here onwards.
    s0 = w[0] * w[0]
    s1 = w[1] * w[1]
    s2 = w[2] * w[2]
    s3 = w[3] * w[3]
    norm = torch.sqrt(s0 + s1 + s2 + s3 + 0.0001)
    w0_n = w[0] / norm
    w1_n = w[1] / norm
    w2_n = w[2] / norm
    w3_n = w[3] / norm

    norm_factor = 2.0
    s1 = norm_factor * (w1_n * w1_n)
    s2 = norm_factor * (w2_n * w2_n)
    s3 = norm_factor * (w3_n * w3_n)
    rot01 = norm_factor * w0_n * w1_n
    rot02 = norm_factor * w0_n * w2_n
    rot03 = norm_factor * w0_n * w3_n
    rot12 = norm_factor * w1_n * w2_n
    rot13 = norm_factor * w1_n * w3_n
    rot23 = norm_factor * w2_n * w3_n

    scale = w[4]
    zero_kernel = w[5]

    k1 = torch.cat(
        [
            zero_kernel,
            scale * (1.0 - (s2 + s3)),
            scale * (rot12 - rot03),
            scale * (rot13 + rot02),
        ],
        dim=1,
    )
    k2 = torch.cat(
        [
            zero_kernel,
            scale * (rot12 + rot03),
            scale * (1.0 - (s1 + s3)),
            scale * (rot23 - rot01),
        ],
        dim=1,
    )
    k3 = torch.cat(
        [
            zero_kernel,
            scale * (rot13 - rot02),
            scale * (rot23 + rot01),
            scale * (1.0 - (s1 + s2)),
        ],
        dim=1,
    )
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k

def get_3d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an octonion kernel.
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)
    assert len(w) == 8

    k0 = torch.cat([w[0], w[1] * g[0], w[2] * g[1], w[3] * g[2], -w[4] * g[0] * g[1], -w[5] * g[0] * g[2], -w[6] * g[1] * g[2], -w[7] * g[0] * g[1] * g[2],], dim=1,)
    k1 = torch.cat([w[1], w[0], -w[4] * g[1], -w[5] * g[2], w[2] * g[1], w[3] * g[2], -w[7] * g[1] * g[2], -w[6] * g[2] * g[1]], dim=1,)
    k2 = torch.cat([w[2], w[4] * g[0], w[0], -w[6] * g[2], -w[1] * g[0], w[7] * g[0] * g[2], w[3] * g[2], w[5] * g[2] * g[0]], dim=1,)
    k3 = torch.cat([w[3], w[5] * g[0], w[6] * g[1], w[0], -w[7] * g[0] * g[1], -w[1] * g[0], -w[2] * g[1], -w[4] * g[0] * g[1]], dim=1,)
    k4 = torch.cat([w[4], w[2], -w[1], g[2] * w[7], w[0], -w[6] * g[2], w[5] * g[2], w[3] * g[2]], dim=1)
    k5 = torch.cat([w[5], w[3], -w[7] * g[1], -w[1], w[6] * g[1], w[0], -w[4] * g[1], -w[2] * g[1]], dim=1)
    k6 = torch.cat([w[6], w[7] * g[0], w[3], -w[2], -w[5] * g[0], w[4] * g[0], w[0], w[1] * g[0]], dim=1)
    k7 = torch.cat([w[7], w[6], -w[5], w[4], w[3], -w[2], w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return 8, k

def get_2d_sandwich_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Sandwich kernel for Clifford algebras of signature CL_{p,q}, p+q=4, g = [-1, -1] corresponds to a quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    g1 = g[0]
    g2 = g[1]
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    w12 = w[3]

    epsilon = 1e-10

    ww_ = g1*g2*w12**2 - g1*w1**2 - g2*w2**2 + w0**2
    _ = -1/F.softplus(torch.abs(ww_))
    

    #ww_tilde0 = g1*g2*w12**2 + g1*w1**2 + g2*w2**2 + w0**2
    
    WRL0 = torch.cat([_*ww_, _*torch.zeros_like(ww_), _*torch.zeros_like(ww_), _*torch.zeros_like(ww_)], dim=1)
    WRL1 = torch.cat([_*torch.zeros_like(ww_), _*(-g1*g2*w12**2 - g1*w1**2 + g2*w2**2 + w0**2), _*(2*g2*(w0*w12 - w1*w2)), _*(2*g2*(g1*w1*w12 - w0*w2))], dim=1)
    WRL2 = torch.cat([_*torch.zeros_like(ww_), _*(2*g1*(-w0*w12 - w1*w2)), _*(-g1*g2*w12**2 + g1*w1**2 - g2*w2**2 + w0**2), _*(2*g1*(g2*w12*w2 + w0*w1))], dim=1)
    WRL3 = torch.cat([_*torch.zeros_like(ww_), _*(-2*g1*w1*w12 - 2*w0*w2), _*(-2*g2*w12*w2 + 2*w0*w1), _*(g1*g2*w12**2 + g1*w1**2 + g2*w2**2 + w0**2)], dim=1)
    _WRL = torch.cat([WRL0, WRL1, WRL2, WRL3], dim=0) # wxw^{-1} = W^{RL}_{w^{-1}w} \underline{x} = \frac{1}{W^{L}_w\overline{w}}W^{RL}_{\overline{w}w} \underline{x}
    return 4, _WRL

def get_3d_sandwich_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Sandwich kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an Cl(0,3) kernel.
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)
    assert len(w) == 8

    g1 = g[0]
    g2 = g[1]
    g3 = g[2]
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    w3 = w[3]
    w12 = w[4]
    w13 = w[5]
    w23 = w[6]
    w123 = w[7]

    epsilon = 1e-7 

    denominator = 4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)**2 + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)**2
    #_ = -1/(denominator + epsilon)
    _ = 1

    WRRL0 = torch.cat([_*denominator, _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator)], dim=1)
    WRRL1 = torch.cat([_*torch.zeros_like(denominator), _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 - w1*w23 + w12*w3 - w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 + g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 - g2*w2**2 - g3*w3**2 - w0**2)), 
                       _*(2*g2*(2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*w12*w123 + g1*w1*w13 - g2*w2*w23 - w0*w3) - (g3*w123*w3 - g3*w13*w23 + w0*w12 - w1*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g3*(2*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g3*w123*w13 - g1*w1*w12 - g3*w23*w3 + w0*w2) - (g2*w12*w23 - g2*w123*w2 + w0*w13 - w1*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g2*(2*g1*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*w12*w23 - g2*w123*w2 + w0*w13 - w1*w3) + (g1*g3*w123*w13 - g1*w1*w12 - g3*w23*w3 + w0*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g3*(-2*g1*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g3*w123*w3 - g3*w13*w23 + w0*w12 - w1*w2) - (g1*g2*w12*w123 + g1*w1*w13 - g2*w2*w23 - w0*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g2*g3*(-(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 + g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 - g2*w2**2 - g3*w3**2 - w0**2) + (w0*w123 - w1*w23 + w12*w3 - w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL2 = torch.cat([_*torch.zeros_like(denominator), _*(2*g1*(-2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*w12*w123 - g1*w1*w13 + g2*w2*w23 - w0*w3) + (g3*w123*w3 + g3*w13*w23 + w0*w12 + w1*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 + w1*w23 + w12*w3 + w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 - g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 + g2*w2**2 - g3*w3**2 - w0**2)), 
                       _*(2*g3*(2*g1*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*g3*w123*w23 - g2*w12*w2 + g3*w13*w3 - w0*w1) - (g1*w1*w123 - g1*w12*w13 + w0*w23 - w2*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g1*(2*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*w1*w123 - g1*w12*w13 + w0*w23 - w2*w3) + (g2*g3*w123*w23 - g2*w12*w2 + g3*w13*w3 - w0*w1)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g1*g3*((w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 - g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 + g2*w2**2 - g3*w3**2 - w0**2) - (w0*w123 + w1*w23 + w12*w3 + w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g3*(-2*g1*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g3*w123*w3 + g3*w13*w23 + w0*w12 + w1*w2) - (g1*g2*w12*w123 - g1*w1*w13 + g2*w2*w23 - w0*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL3 = torch.cat([_*torch.zeros_like(denominator), _*(2*g1*(-2*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g3*w123*w13 + g1*w1*w12 + g3*w23*w3 + w0*w2) - (g2*w12*w23 + g2*w123*w2 - w0*w13 - w1*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g2*(-2*g1*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*g3*w123*w23 + g2*w12*w2 - g3*w13*w3 - w0*w1) + (g1*w1*w123 + g1*w12*w13 + w0*w23 + w2*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 + w1*w23 - w12*w3 - w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 + g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 - g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g1*g2*(-(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 + g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 - g2*w2**2 + g3*w3**2 - w0**2) + (w0*w123 + w1*w23 - w12*w3 - w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g1*(2*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*w1*w123 + g1*w12*w13 + w0*w23 + w2*w3) + (g2*g3*w123*w23 + g2*w12*w2 - g3*w13*w3 - w0*w1)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g2*(2*g1*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*w12*w23 + g2*w123*w2 - w0*w13 - w1*w3) - (g1*g3*w123*w13 + g1*w1*w12 + g3*w23*w3 + w0*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL4 = torch.cat([_*torch.zeros_like(denominator), _*(-4*g1*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*w12*w23 + g2*w123*w2 - w0*w13 - w1*w3) + 2*(g1*g3*w123*w13 + g1*w1*w12 + g3*w23*w3 + w0*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(4*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*w1*w123 + g1*w12*w13 + w0*w23 + w2*w3) + 2*(g2*g3*w123*w23 + g2*w12*w2 - g3*w13*w3 - w0*w1)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g3*((w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 + g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 - g2*w2**2 + g3*w3**2 - w0**2) - (w0*w123 + w1*w23 - w12*w3 - w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 + w1*w23 - w12*w3 - w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 + g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 - g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g3*(2*g1*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*g3*w123*w23 + g2*w12*w2 - g3*w13*w3 - w0*w1) - (g1*w1*w123 + g1*w12*w13 + w0*w23 + w2*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g3*(-2*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g3*w123*w13 + g1*w1*w12 + g3*w23*w3 + w0*w2) - (g2*w12*w23 + g2*w123*w2 - w0*w13 - w1*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL5 = torch.cat([_*torch.zeros_like(denominator), _*(-4*g1*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g3*w123*w3 + g3*w13*w23 + w0*w12 + w1*w2) - 2*(g1*g2*w12*w123 - g1*w1*w13 + g2*w2*w23 - w0*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g2*(-(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 - g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 + g2*w2**2 - g3*w3**2 - w0**2) + (w0*w123 + w1*w23 + w12*w3 + w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*w1*w123 - g1*w12*w13 + w0*w23 - w2*w3) + 2*(g2*g3*w123*w23 - g2*w12*w2 + g3*w13*w3 - w0*w1)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g2*(-2*g1*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*g3*w123*w23 - g2*w12*w2 + g3*w13*w3 - w0*w1) + (g1*w1*w123 - g1*w12*w13 + w0*w23 - w2*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 + w1*w23 + w12*w3 + w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 - g1*g3*w13**2 - g1*w1**2 + g2*g3*w23**2 + g2*w2**2 - g3*w3**2 - w0**2)), 
                       _*(2*g2*(2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*w12*w123 - g1*w1*w13 + g2*w2*w23 - w0*w3) - (g3*w123*w3 + g3*w13*w23 + w0*w12 + w1*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL6 = torch.cat([_*torch.zeros_like(denominator), _*(2*g1*((w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 + g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 - g2*w2**2 - g3*w3**2 - w0**2) - (w0*w123 - w1*w23 + w12*w3 - w13*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(-4*g1*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g3*w123*w3 - g3*w13*w23 + w0*w12 - w1*w2) - 2*(g1*g2*w12*w123 + g1*w1*w13 - g2*w2*w23 - w0*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(-4*g1*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g2*w12*w23 - g2*w123*w2 + w0*w13 - w1*w3) - 2*(g1*g3*w123*w13 - g1*w1*w12 - g3*w23*w3 + w0*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)), 
                       _*(2*g1*(2*g2*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g3*w123*w13 - g1*w1*w12 - g3*w23*w3 + w0*w2) - (g2*w12*w23 - g2*w123*w2 + w0*w13 - w1*w3)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(2*g1*(-2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(g1*g2*w12*w123 + g1*w1*w13 - g2*w2*w23 - w0*w3) + (g3*w123*w3 - g3*w13*w23 + w0*w12 - w1*w2)*(g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2))), 
                       _*(4*g1*g2*g3*(w0*w123 - w1*w23 - w12*w3 + w13*w2)*(w0*w123 - w1*w23 + w12*w3 - w13*w2) + (g1*g2*g3*w123**2 - g1*g2*w12**2 - g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 + g2*w2**2 + g3*w3**2 - w0**2)*(g1*g2*g3*w123**2 + g1*g2*w12**2 + g1*g3*w13**2 + g1*w1**2 - g2*g3*w23**2 - g2*w2**2 - g3*w3**2 - w0**2)), _*torch.zeros_like(denominator)], dim=1)
    
    WRRL7 = torch.cat([_*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*torch.zeros_like(denominator), _*denominator], dim=1)
    _WRRL = torch.cat([WRRL0, WRRL1, WRRL2, WRRL3, WRRL4, WRRL5, WRRL6, WRRL7], dim=0)
    return 8, _WRRL

def clifford_convnd(conv_fn: Callable, x: torch.Tensor, output_blades: int, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs,
                    ) -> torch.Tensor:
    """Apply a Clifford convolution to a tensor.
    Args:
        conv_fn (Callable): The convolution function to use.
        x (torch.Tensor): Input tensor.
        output_blades (int): The output blades of the Clifford algebra.
        Different from the default n_blades when using encoding and decoding layers.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
    Returns:
        torch.Tensor: Convolved output tensor.
    """
    # Reshape x such that the convolution function can be applied.
    #print('x shape: '+str(x.shape))
    B, *_ = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    x = x.permute(B_dim, -1, C_dim, *D_dims)
    x = x.reshape(B, -1, *x.shape[3:])
    # Apply convolution function
    output = conv_fn(x, weight, bias=bias, **kwargs)
    #print('\n weights shape: '+str(weight.shape))
    # Reshape back.
    output = output.view(B, output_blades, -1, *output.shape[2:])
    B_dim, I_dim, C_dim, *D_dims = range(len(output.shape))
    output = output.permute(B_dim, C_dim, *D_dims, I_dim)
    return output


class _CliffordConvNd(nn.Module):
    """Base class for all Clifford convolution modules."""

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int, dilation: int, groups: int, bias: bool, padding_mode: str, sandwich: bool = False, ) -> None:
        super().__init__()
        sig = CliffordSignature(g)
        # register as buffer as we want the tensor to be moved to the same device as the module
        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        #if self.dim == 1:
        #    self._get_kernel = get_1d_clifford_kernel
        if self.dim == 2 and sandwich:
            self._get_kernel = get_2d_sandwich_kernel 
        elif self.dim == 3 and sandwich:
            self._get_kernel = get_3d_sandwich_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel #get_2d_sandwich_kernel #get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel #get_3d_sandwich_kernel #get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford convolution not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        if padding_mode != "zeros":
            raise NotImplementedError(f"Padding mode {padding_mode} not implemented.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.sandwich = sandwich

        #print('\n in_channels : '+str(self.in_channels)+' out_channels: '+str(self.out_channels)+' kernel_size: '+str(self.kernel_size)+' groups:'+str(self.groups))

        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size)) for _ in range(self.n_blades)]
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        
        #print('\n weights: '+str(self.weight))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of the Clifford convolution weight and bias tensors.
        The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        """
        for blade, w in enumerate(self.weight):
            # Weight initialization for Clifford weights.
            if blade < self.n_blades:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    torch.Tensor(
                        self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size
                    )
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(w, -bound, bound)
            elif blade == self.n_blades + 1:
                # Nothing to be done for zero kernel.
                pass
            else:
                raise ValueError(
                    f"Wrong number of Clifford weights. Expected {self.n_blades} weight tensors, and 2 extra tensors for rotational kernels."
                )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                torch.Tensor(self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, conv_fn: callable) -> torch.Tensor:
        if self.bias is not None:
            b = self.bias.view(-1)
        else:
            b = None
        output_blades, w = self._get_kernel(self.weight, self.g)
        return clifford_convnd(conv_fn, x, output_blades, w, b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"

        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"

        return s.format(**self.__dict__)

class CliffordConv1d(_CliffordConvNd):
    """1d Clifford convolution.

    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
    """

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", sandwich: bool = False,) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)

        super().__init__(g, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode, sandwich,)
        #if not self.dim == 2:
        #    raise NotImplementedError("Wrong Clifford signature for CliffordConv1d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        return super().forward(x, F.conv1d)

class CliffordConv2d(_CliffordConvNd):
    """2d Clifford convolution (dim(g)=2).
    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): padding added to both sides of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
        sandwich (bool): If True, enables the sandwich kernel for Clifford convolution.
    """

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", sandwich: bool = False,):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(g, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode, sandwich,)
        if not self.dim == 2 and not self.dim == 3:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv2d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        return super().forward(x, F.conv2d)

class CliffordConv3d(_CliffordConvNd):
    """3d Clifford convolution.
    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int, int]]): padding added to all sides of the input.
        dilation (Union[int, Tuple[int, int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
    """

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", sandwich: bool = False,):
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)

        super().__init__(g, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode, sandwich,)
        if not self.dim == 2 and not self.dim == 3:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv3d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        return super().forward(x, F.conv3d)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CliffordLinear(nn.Module):
    """Clifford linear layer.
    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """

    def __init__(self, g, in_channels: int, out_channels: int, bias: bool = True, sandwich: bool = False) -> None:
        super().__init__()
        sig = CliffordSignature(g)

        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades

        if self.dim == 2 and sandwich:
            self._get_kernel = get_2d_sandwich_kernel 
        elif self.dim == 3 and sandwich:
            self._get_kernel = get_3d_sandwich_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Custom initialization of the Clifford linear weight and bias tensors.
        stdv = 1.0 / math.sqrt(self.weight.numel())
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(self.bias.numel())
            self.bias.data.uniform_(-bound, bound)

        # Convert weights and bias to float16
        #self.weight.data = self.weight.data.to(torch.float16)
        #if self.bias is not None:
        #    self.bias.data = self.bias.data.to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that the Clifford kernel can be applied.
        B, _, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        B_dim, C_dim, I_dim = range(len(x.shape))
        x = x.permute(B_dim, -1, C_dim)
        x = x.reshape(B, -1)
        # Get Clifford kernel, apply it.
        _, weight = self._get_kernel(self.weight, self.g)
        if self.bias is not None:
            output = F.linear(x, weight, self.bias.view(-1))
        else:
            output = F.linear(x, weight)

        # Check for NaN or Inf in weights and biases
        if torch.isnan(weight).any() or torch.isinf(weight).any():
            raise ValueError("Weights contain NaN or Inf values after forward pass.")
        if self.bias is not None and (torch.isnan(self.bias).any() or torch.isinf(self.bias).any()):
            raise ValueError("Bias contains NaN or Inf values after forward pass.")

        # Reshape back.
        output = output.view(B, I, -1)
        B_dim, I_dim, C_dim = range(len(output.shape))
        output = output.permute(B_dim, C_dim, I_dim)

        torch.cuda.empty_cache()
        return output


def whiten_data(x: torch.Tensor, training: bool = True, running_mean: Optional[torch.Tensor] = None, running_cov: Optional[torch.Tensor] = None,
                momentum: float = 0.1, eps: float = 1e-4,) -> torch.Tensor:
    """Jointly whiten features in tensors `(B, C, *D, I)`: take n_blades(I)-dim vectors
    and whiten individually for each channel dimension C over `(B, *D)`.
    I is the number of blades in the respective Clifford algebra, e.g. I = 2 for complex numbers.
    Args:
        x (torch.Tensor): The tensor to whiten.
        training (bool, optional): Wheter to update the running mean and covariance. Defaults to `True`.
        running_mean (torch.Tensor, optional): The running mean of shape `(I, C). Defaults to `None`.
        running_cov (torch.Tensor, optional): The running covariance of shape `(I, I, C)` Defaults to `None`.
        momentum (float, optional): The momentum to use for the running mean and covariance. Defaults to `0.1`.
        eps (float, optional): A small number to add to the covariance. Defaults to 1e-5.

    Returns:
        (torch.Tensor): Whitened data of shape `(B, C, *D, I)`.
    """
    assert x.dim() >= 3
    # Get whitening shape of [1, C, ...]
    _, C, *_, I = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    shape = 1, C, *([1] * (x.dim() - 3))

    # Get feature mean.
    if not (running_mean is None or running_mean.shape == (I, C)):
        raise ValueError(f"Running_mean expected to be none, or of shape ({I}, {C}).")
    if training or running_mean is None:
        mean = x.mean(dim=(B_dim, *D_dims))
        if running_mean is not None:
            running_mean += momentum * (mean.data.permute(1, 0) - running_mean)
    else:
        mean = running_mean.permute(1, 0)

    # Get feature covariance.
    x = x - mean.reshape(*shape, I)
    if not (running_cov is None or running_cov.shape == (I, I, C)):
        raise ValueError(f"Running_cov expected to be none, or of shape ({I}, {I}, {C}).")
    if training or running_cov is None:
        # B, C, *D, I -> C, I, B, *D
        X = x.permute(C_dim, I_dim, B_dim, *D_dims).flatten(2, -1)
        # Covariance XX^T matrix of shape C x I x I
        cov = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]
        if running_cov is not None:
            running_cov += momentum * (cov.data.permute(1, 2, 0) - running_cov)

    else:
        cov = running_cov.permute(2, 0, 1)

    # Upper triangle Cholesky decomposition of covariance matrix: U^T U = Cov
    eye = eps * torch.eye(I, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov = cov + eye

    if cov.dtype == torch.float16:
        cov = cov.float()
    
    U = torch.linalg.cholesky(cov).mH
    # Invert Cholesky decomposition, returns tensor of shape [B, C, *D, I]
    x_whiten = torch.linalg.solve_triangular(U.reshape(*shape, I, I), x.unsqueeze(-1), upper=True,).squeeze(-1)
    return x_whiten

def clifford_batch_norm(x: torch.Tensor, n_blades: int, running_mean: Optional[torch.Tensor] = None, running_cov: Optional[torch.Tensor] = None,
                        weight: Optional[Union[torch.Tensor, nn.Parameter]] = None, bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,
                        training: bool = True, momentum: float = 0.1, eps: float = 1e-05,) -> torch.Tensor:
    """Clifford batch normalization for each channel across a batch of data.
    Args:
        x (torch.Tensor): Input tensor of shape `(B, C, *D, I)` where I is the blade of the algebra.
        n_blades (int): Number of blades of the Clifford algebra.
        running_mean (torch.Tensor, optional): The tensor with running mean statistics having shape `(I, C)`.
        running_cov (torch.Tensor, optional): The tensor with running covariance statistics having shape `(I, I, C)`.
        weight (Union[torch.Tensor, nn.Parameter], optional): Additional weight tensor which is applied post normalization, and has the shape `(I, I, C)`.
        bias (Union[torch.Tensor, nn.Parameter], optional): Additional bias tensor which is applied post normalization, and has the shape `(I, C)`.
        training (bool, optional): Whether to use the running mean and variance. Defaults to True. Defaults to True.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to 0.1.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to 1e-05.
    Returns:
        (torch.Tensor): Normalized input of shape `(B, C, *D, I)`
    """
    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    # Whiten and apply affine transformation
    _, C, *_, I = x.shape
    assert I == n_blades
    x_norm = whiten_data(x, training=training, running_mean=running_mean, running_cov=running_cov, momentum=momentum, eps=eps,)
    #x_norm = x
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (I, I, C)
        assert bias.shape == (I, C)
        # Unsqueeze weight and bias for each dimension except the channel dimension.
        shape = 1, C, *([1] * (x.dim() - 3))
        weight = weight.reshape(I, I, *shape)
        # Apply additional affine transformation post normalization.
        weight_idx = list(range(weight.dim()))
        # TODO: weight multiplication should be changed to geometric product.
        weight = weight.permute(*weight_idx[2:], *weight_idx[:2])
        x_norm = weight.matmul(x_norm[..., None]).squeeze(-1) + bias.reshape(*shape, I)

    return x_norm

def clifford_group_norm(x: torch.Tensor, n_blades: int, num_groups: int = 1, running_mean: Optional[torch.Tensor] = None,
                        running_cov: Optional[torch.Tensor] = None, weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
                        bias: Optional[Union[torch.Tensor, nn.Parameter]] = None, training: bool = True, momentum: float = 0.1,
                        eps: float = 1e-05,) -> torch.Tensor:
    """Clifford group normalization
    Args:
        x (torch.Tensor): Input tensor of shape `(B, C, *D, I)` where I is the blade of the algebra.

        n_blades (int): Number of blades of the Clifford algebra.

        num_groups (int): Number of groups for which normalization is calculated. Defaults to 1.
                          For `num_groups == 1`, it effectively applies Clifford layer normalization, for `num_groups == C`, it effectively applies Clifford instance normalization.

        running_mean (torch.Tensor, optional): The tensor with running mean statistics having shape `(I, C / num_groups)`. Defaults to None.
        running_cov (torch.Tensor, optional): The tensor with running real-imaginary covariance statistics having shape `(I, I, C / num_groups)`. Defaults to None.

        weight (Union[torch.Tensor, nn.Parameter], optional): Additional weight tensor which is applied post normalization, and has the shape `(I, I, C / num_groups)`. Defaults to None.

        bias (Union[torch.Tensor, nn.Parameter], optional): Additional bias tensor which is applied post normalization, and has the shape `(I, C / num_groups)`. Defaults to None.

        training (bool, optional): Whether to use the running mean and variance. Defaults to True.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to 0.1.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to 1e-05.
    Returns:
        (torch.Tensor): Group normalized input of shape `(B, C, *D, I)`.
    """

    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    B, C, *D, I = x.shape
    assert num_groups <= C
    assert C % num_groups == 0, "Number of channels should be evenly divisible by the number of groups."
    assert I == n_blades
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (I, I, int(C / num_groups))
        assert bias.shape == (I, int(C / num_groups))
        weight = weight.repeat(1, 1, B)
        bias = bias.repeat(1, B)

    def _instance_norm(x, num_groups, running_mean, running_cov, weight, bias, training, momentum, eps,):
        if running_mean is not None and running_cov is not None:
            assert running_mean.shape == (I, int(C / num_groups))
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(1, B)
            assert running_cov.shape == (I, I, int(C / num_groups))
            running_cov_orig = running_cov
            running_cov = running_cov_orig.repeat(1, 1, B)

        # Reshape such that batch normalization can be applied.
        # For num_groups == 1, it defaults to layer normalization,
        # for num_groups == C, it defaults to instance normalization.
        x_reshaped = x.reshape(1, int(B * C / num_groups), num_groups, *D, I)

        x_norm = clifford_batch_norm(x_reshaped, n_blades, running_mean, running_cov, weight, bias, training, momentum, eps,)

        # Reshape back running mean and running var.
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(I, B, int(C / num_groups)).mean(1, keepdim=False))
        if running_cov is not None:
            running_cov_orig.copy_(running_cov.view(I, I, B, int(C / num_groups)).mean(1, keepdim=False))

        return x_norm.view(B, C, *D, I)

    return _instance_norm(x, num_groups, running_mean, running_cov, weight, bias, training, momentum, eps,)

class _CliffordBatchNorm(nn.Module):
    def __init__(self, g: Union[tuple, list, torch.Tensor], channels: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                track_running_stats: bool = True,):
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(self.n_blades, self.n_blades, channels))
            self.bias = torch.nn.Parameter(torch.empty(self.n_blades, channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(self.n_blades, channels))
            self.register_buffer("running_cov", torch.empty(self.n_blades, self.n_blades, channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_cov", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_running_stats()
        self.reset_parameters()

    def reset_running_stats(self):
        if not self.track_running_stats:
            return

        self.num_batches_tracked.zero_()
        self.running_mean.zero_()
        self.running_cov.copy_(torch.eye(self.n_blades, self.n_blades).unsqueeze(-1))

    def reset_parameters(self):
        if not self.affine:
            return

        self.weight.data.copy_(torch.eye(self.n_blades, self.n_blades).unsqueeze(-1))
        init.zeros_(self.bias)

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    # Use cumulative moving average.
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    # Use exponential moving average.
                    exponential_average_factor = self.momentum

        return clifford_batch_norm(x, self.n_blades, self.running_mean, self.running_cov, self.weight, self.bias,
                                    self.training or not self.track_running_stats, exponential_average_factor,
                                    self.eps,)

    def extra_repr(self):
        return (
            "{channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )

class _CliffordGroupNorm(_CliffordBatchNorm):
    def __init__(self, g: Union[tuple, list, torch.Tensor], num_groups: int, channels: int, eps: float = 1e-5, momentum: float = 0.1,
                affine: bool = True, track_running_stats: bool = False,):
        self.num_groups = num_groups
        super().__init__(g, int(channels / num_groups), eps, momentum, affine, track_running_stats=track_running_stats,)

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return clifford_group_norm(x, self.n_blades, self.num_groups, self.running_mean, self.running_cov, self.weight, self.bias, 
                                   self.training or not self.track_running_stats, exponential_average_factor, self.eps,)

    def extra_repr(self):
        return (
            "{num_groups}, {channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )
    
class CliffordGroupNorm1d(_CliffordGroupNorm):
    """Clifford group normalization for 2D or 3D data.
    The input data is expected to be at least 3d, with shape `(B, C, D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension (if present).
    """
    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 3 and x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")

class CliffordGroupNorm2d(_CliffordGroupNorm):
    """Clifford group normalization for 4D data.
    The input data is expected to be 4D, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 2 dimensions.
    """
    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 5:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")

class CliffordGroupNorm3d(_CliffordGroupNorm):
    """Clifford group normalization for 4D data.
    The input data is expected to be 5D, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 3 dimensions.
    """
    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 6:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")

class CliffordSumVSiLU(nn.Module):
    """
    A module that applies the vector SiLU using vector sum to vectors in Cl(p,q).
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(input.sum(-1, keepdim=True)) * input
    
class IdentityActivation(nn.Module):
    """
    A module that applies the identity activation in Cl(p,q).
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

def clean_cache_directory():
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    csv_dir = os.path.join(cache_dir, "csv")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(csv_dir)


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2CliffordHead(nn.Module):
    """Clifford Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.g = config.g
        self.num_blades = 2**len(self.g)
        self.blades_idxs = config.blades_idxs
        self.sandwich = False
        self.norm = True

        #print(config.hidden_size)

        #self.batch_norm1 = CliffordGroupNorm1d(self.g, 1, config.hidden_size) if self.norm else nn.Identity()
        #self.batch_norm2 = CliffordGroupNorm1d(self.g, 1, config.dense_1_out) if self.norm else nn.Identity()

        self.activation = F.leaky_relu
        #self.activation = CliffordSumVSiLU()
        #self.activation = torch.tanh

        self.dense1 = CliffordLinear(self.g, config.hidden_size, config.dense_1_out, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, config.dense_1_out, config.num_labels, bias=False, sandwich=self.sandwich)
        self.dense3 = nn.Linear(self.num_blades, 1, bias=True)

        self.dropout_1 = nn.Dropout(config.dropout_1)
        self.dropout_2 = nn.Dropout(config.dropout_1)

    def forward(self, features, **kwargs):

        if len(features.size()) == 1:
            B = 1
            C = features.size()[0]
        else:
            B, C = features.size()

        multivectors = torch.zeros(B, C, self.num_blades).to(self.device)
        multivectors[:, :, self.blades_idxs] = features.reshape(B, C, 1)

        B, C, Q = multivectors.size()
        #multivectors = multivectors.reshape(B, C, Q)

        d1 = self.activation(self.dense1(multivectors))
        del multivectors

        B, C_, Q = d1.size()
        d1 = d1.reshape(B, C_*Q)
        d1 = self.dropout_1(d1)
        d1 = d1.reshape(B, C_, Q)

        d2 = self.dense2(d1)
        del d1

        B, C_, Q = d2.size()
        d2 = d2.reshape(B, C_*Q)
        d2 = self.dropout_2(d2)
        d2 = d2.reshape(B*C_, Q)

        out = self.dense3(d2)
        del d2

        out = out.view(B, C_)

        torch.cuda.empty_cache()

        return out

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
        #self.classifier = Wav2Vec2ClassificationHead(config)
        self.classifier = Wav2Vec2CliffordHead(config)

        self.init_weights()

    def summary_classifier(self):
        summary(self.classifier, input_size=(self.config.hidden_size,))

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
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

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)
            #print(f"GPU memory allocated after wav2vec2: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        hidden_states = outputs[0]
        #print('\nembedding pre pool: '+str(hidden_states))
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        #print('\nembedding post pool: '+str(hidden_states))
        logits = self.classifier(hidden_states)

        del hidden_states

        #print(f"GPU memory allocated after classifier: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        #print('\npredictions: '+str(logits))

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
        
        #print('loss function:'+str(loss_fct))
        #print('\nloss value'+str(loss))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,)
    
@dataclass
class DataCollatorCTCWithPadding:

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

        batch = self.processor.pad(input_features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",)

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        del input_features
        del label_features

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
            setattr(self.config, 'g', hyperparameters['g'])
            setattr(self.config, 'blades_idxs', hyperparameters['blades_idxs'])
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
        if 'g' in hyperparameters:
            setattr(self.config, 'g', hyperparameters['g'])
        if 'blades_idxs' in hyperparameters:
            setattr(self.config, 'blades_idxs', hyperparameters['blades_idxs'])

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

    
        clean_cache_directory()
        if (os.path.exists(f"{save_path_csv}/train.csv") and os.path.exists(f"{save_path_csv}/eval.csv") and os.path.exists(f"{save_path_csv}/test.csv")) == False:
            if augmentation:
                dataset_wav2Vec_train = augmentation_wav2vec(dataset_wav2Vec_train, 0.1, "train") # just here
                dataset_wav2Vec_eval = augmentation_wav2vec(dataset_wav2Vec_eval, 0.1, "eval") # no here
                dataset_wav2Vec_test = augmentation_wav2vec(dataset_wav2Vec_test, 0.1, "test") # no here
            # train eval 
            train_df = dataset_wav2Vec_train.reset_index(drop=True)
            eval_df = dataset_wav2Vec_eval.reset_index(drop=True)
            #print('\n '+save_path_csv+'\n')
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

        #print("Train dataset: ", len(train))
        #print("Eval dataset: ", len(eval))
        #print("Test dataset: ", len(test))

        #print(pd.Series(train['Emotion']).value_counts())
        #print(pd.Series(eval['Emotion']).value_counts())
        #print(pd.Series(test['Emotion']).value_counts())
        
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        target_sampling_rate = processor.feature_extractor.sampling_rate
        #print(f"The target sampling rate: {target_sampling_rate}")

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
        training_args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2,
                                        evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=30, fp16=True, logging_strategy="epoch",
                                        learning_rate=1e-4, save_total_limit=1, metric_for_best_model="eval_accuracy", load_best_model_at_end=True,)

        # prepare Trainer
        trainer = Trainer(model=model, data_collator=data_collator, args=training_args, compute_metrics=compute_metrics, train_dataset=train_dataset,
                        eval_dataset=eval_dataset, tokenizer=processor.feature_extractor, callbacks=[EarlyStoppingCallback(early_stopping_patience=5), self.mycallbacks])

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

