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
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn import init
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Union

seed = 42
torch.manual_seed(seed)

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


class CliffordLinear(nn.Module):
    """Clifford linear layer.
    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """

    def __init__(self, g, in_channels: int, out_channels: int, bias: bool = True, sandwich: bool = False,) -> None:
        super().__init__()
        sig = CliffordSignature(g)

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
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        # need to modify this in order to use sendwich kernels OLD: self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(out_channels, in_channels)) for _ in range(self.n_blades)]
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        # need to modify this in order to use sendwich kernels. OLD:
        # nn.init.kaiming_uniform_(
        #    self.weight.view(self.out_channels, self.in_channels * self.n_blades),
        #    a=math.sqrt(5),
        # )
        # if self.bias is not None:
        #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
        #        self.weight.view(self.out_channels, self.in_channels * self.n_blades)
        #    )
        #    bound = 1 / math.sqrt(fan_in)
        #    nn.init.uniform_(self.bias, -bound, bound)
        for blade, w in enumerate(self.weight):
            # Weight initialization for Clifford weights.
            if blade < self.n_blades:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    torch.Tensor(
                        self.out_channels, int(self.in_channels * self.n_blades))
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(w, -bound, bound)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                torch.Tensor(self.out_channels, int(self.in_channels * self.n_blades))
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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
        # Reshape back.
        output = output.view(B, I, -1)
        B_dim, I_dim, C_dim = range(len(output.shape))
        output = output.permute(B_dim, C_dim, I_dim)
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

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return (x / xsum * xshape[2] * xshape[3] * 0.5)

class CliffCNN1D(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffCNN1D, self).__init__()
        self.conv_in_channels = 1
        self.conv_hidden_channels = model_params['dense_units'][0]
        self.conv_out_channels = model_params['dense_units'][1]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = model_params['kernel_size']//2

        self.dense_init_dim = self.update_dim(self.update_dim(128))
        self.dense_init_units = model_params['dense_units'][2]
        self.dense_units_in_channels = self.dense_init_dim * self.dense_init_units
        self.dense_units_hidden_channels = model_params['dense_units'][3]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.2

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        if self.conv_hidden_channels==1 and self.conv_out_channels==1 and self.dense_init_units==1 :
            self.activation = IdentityActivation()
        else:
            self.activation = CliffordSumVSiLU()

        self.conv1 = CliffordConv1d(self.g, self.conv_in_channels, self.conv_hidden_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv2 = CliffordConv1d(self.g, self.conv_hidden_channels, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.dense_units_in_channels, self.dense_units_hidden_channels, bias=True, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels, self.dense_units_out_channels, bias=True, sandwich=self.sandwich)
        self.dense3 = nn.Linear(self.num_blades, 1, bias=True)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateDense)
        self.dropout_4 = nn.Dropout(self.dropout_rateDense)
    
    def update_dim(self, S):
        if S%2==0:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride)+1))
        else:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
    
    def forward(self, inputs):
        
        B, S, _ = inputs.size()
        C = len(self.blades_idxs)

        multivectors = torch.zeros(B, S//C, self.num_blades)
        multivectors[:, :, self.blades_idxs] = inputs.reshape(B, S//C, C) #if C>1 else inputs.squeeze()

        B, S, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, S, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _S, Q = c1.size()
        #_S = self.update_dim(S)
        c1 = c1.reshape(B, C1*Q, _S)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _S, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, __S, Q = c2.size()
        #__S = self.update_dim(_S)
        c2 = c2.reshape(B, C2*Q, __S)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2*__S, Q)
        
        d1 = self.activation(self.dense1(c2))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_3(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S__, Q = d2.size()
        d2 = d2.reshape(B, Q, S__)
        d2 = self.dropout_4(d2)
        d2 = d2.reshape(B*S__, Q)

        out = self.dense3(d2)
        out = out.view(B, S__)

        return out

class CliffSER1D(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffSER1D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = 0
        #self.pooling_size = model_params['pooling_size']

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        #if self.conv_hidden_channels==1 and  self.conv_out_channels==1 and self.dense_init_units==1 :
        #   self.activation = IdentityActivation()
        #else:
        #   self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        self.activation = F.leaky_relu

        self.conv1 = CliffordConv1d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size,  
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv1d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size,  
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv1d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv1d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size,
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.conv_out_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense4 = nn.Linear(self.num_blades, 1, bias=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
    
    def forward(self, inputs):

        B, S, _ = inputs.size()

        multivectors = torch.zeros(B, S, self.num_blades).to(self.device)
        multivectors[:, :, self.blades_idxs] = inputs.reshape(B, S, 1)

        B, S, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, S, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _S, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _S)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _S, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _S, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _S)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _S, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _S, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _S)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _S, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _S, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _S)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        c4 = self.global_avg_pool(c4)
        c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d3 = d3.reshape(B, S_, Q)
        d3 = d3.reshape(B*S_, Q)

        #out = d3[:, :, self.blades_idxs[1]].squeeze()

        out = self.dense4(d3)
        out = out.view(B, S_)

        return out

class PureCliffSER1D(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(PureCliffSER1D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = model_params['kernel_size']//2

        self.dense_init_dim1 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(128))))
        #self.dense_init_dim1 = self.dense_init_dim1//self.pooling_size**2
        self.dense_units_in_channels = self.dense_init_dim1 * self.conv_out_channels

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_hidden_channels3 = model_params['dense_units'][6]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        if self.conv_hidden_channels1==1:
           self.activation = IdentityActivation()
        else:
           self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        #self.activation = F.silu #F.leaky_relu

        self.conv1 = CliffordConv1d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv1d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv1d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv1d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size,
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.dense_units_in_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_hidden_channels3, bias=False, sandwich=self.sandwich)
        self.dense4 = CliffordLinear(self.g, self.dense_units_hidden_channels3, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense5 = nn.Linear(self.num_blades, 1, bias=True)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
        self.dropout_8 = nn.Dropout(self.dropout_rateDense)
    
    def update_dim(self, S):
        if S%2==0:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride)+1))
        else:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
    
    def forward(self, inputs):

        B, S, _ = inputs.size()

        multivectors = torch.zeros(B, S, self.num_blades).to(self.device)
        multivectors[:, :, self.blades_idxs] = inputs.reshape(B, S, 1)

        B, S, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, S, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _S, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _S)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _S, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _S, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _S)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _S, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _S, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _S)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _S, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _S, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _S)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        c4 = c4.reshape(B, C4*_S, Q)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        #c4 = self.global_avg_pool(c4)
        #c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d2 = d2.reshape(B, S_, Q)
        d3 = d3.reshape(B, S_, Q)

        d4 = self.activation(self.dense4(d3))
        B, S_, Q = d4.size()
        d4 = d4.reshape(B, Q, S_)
        d4 = self.dropout_8(d4)
        #d2 = d2.reshape(B, S__, Q)
        d4 = d4.reshape(B*S_, Q)

        #out = d1[:, :, 0].squeeze()

        out = self.dense5(d4)
        out = out.view(B, S_)

        return out

class CliffSER1D_EMD(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffSER1D_EMD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = 0
        #self.pooling_size = model_params['pooling_size']

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        #if self.conv_hidden_channels==1 and  self.conv_out_channels==1 and self.dense_init_units==1 :
        #   self.activation = IdentityActivation()
        #else:
        #   self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        self.activation = F.leaky_relu

        self.conv1 = CliffordConv1d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size,  
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv1d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size,  
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv1d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv1d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size,
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.conv_out_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense4 = nn.Linear(self.num_blades, 1, bias=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
    
    def forward(self, inputs):

        B, S, C = inputs.size()

        multivectors = torch.zeros(B, S, self.num_blades).to(self.device)
        multivectors[:, :, self.blades_idxs[0]] = inputs.reshape(B, S, C)[...,3] #if C>1 else inputs.squeeze()
        multivectors[:, :, self.blades_idxs[1]] = inputs.reshape(B, S, C)[...,0]
        multivectors[:, :, self.blades_idxs[2]] = inputs.reshape(B, S, C)[...,1]
        multivectors[:, :, self.blades_idxs[3]] = inputs.reshape(B, S, C)[...,2]

        B, S, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, S, Q)

        #import code; code.interact(local=locals()) 

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _S, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _S)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _S, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _S, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _S)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _S, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _S, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _S)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _S, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _S, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _S)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        c4 = self.global_avg_pool(c4)
        c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d3 = d3.reshape(B, S_, Q)
        d3 = d3.reshape(B*S_, Q)

        #out = d3[:, :, self.blades_idxs[1]].squeeze()

        out = self.dense4(d3)
        out = out.view(B, S_)

        return out

class PureCliffSER1D_EMD(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(PureCliffSER1D_EMD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = model_params['kernel_size']//2

        self.dense_init_dim1 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(128))))
        #self.dense_init_dim1 = self.dense_init_dim1//self.pooling_size**2
        self.dense_units_in_channels = self.dense_init_dim1 * self.conv_out_channels

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_hidden_channels3 = model_params['dense_units'][6]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        if self.conv_hidden_channels1==1:
           self.activation = IdentityActivation()
        else:
           self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        #self.activation = F.silu #F.leaky_relu

        self.conv1 = CliffordConv1d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv1d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv1d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv1d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size,
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.dense_units_in_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_hidden_channels3, bias=False, sandwich=self.sandwich)
        self.dense4 = CliffordLinear(self.g, self.dense_units_hidden_channels3, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense5 = nn.Linear(self.num_blades, 1, bias=True)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
        self.dropout_8 = nn.Dropout(self.dropout_rateDense)
    
    def update_dim(self, S):
        if S%2==0:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride)+1))
        else:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
    
    def forward(self, inputs):

        B, S, C = inputs.size()

        multivectors = torch.zeros(B, S, self.num_blades).to(self.device)
        multivectors[:, :, self.blades_idxs[0]] = inputs.reshape(B, S, C)[...,3] #if C>1 else inputs.squeeze()
        multivectors[:, :, self.blades_idxs[1]] = inputs.reshape(B, S, C)[...,0]
        multivectors[:, :, self.blades_idxs[2]] = inputs.reshape(B, S, C)[...,1]
        multivectors[:, :, self.blades_idxs[3]] = inputs.reshape(B, S, C)[...,2]

        B, S, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, S, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _S, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _S)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _S, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _S, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _S)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _S, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _S, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _S)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _S, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _S, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _S)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        c4 = c4.reshape(B, C4*_S, Q)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        #c4 = self.global_avg_pool(c4)
        #c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d2 = d2.reshape(B, S_, Q)
        d3 = d3.reshape(B, S_, Q)

        d4 = self.activation(self.dense4(d3))
        B, S_, Q = d4.size()
        d4 = d4.reshape(B, Q, S_)
        d4 = self.dropout_8(d4)
        #d2 = d2.reshape(B, S__, Q)
        d4 = d4.reshape(B*S_, Q)

        #out = d1[:, :, 0].squeeze()

        out = self.dense5(d4)
        out = out.view(B, S_)

        return out

class CliffSER2D(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffSER2D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = 0
        #self.pooling_size = model_params['pooling_size']

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        #if self.conv_hidden_channels==1 and  self.conv_out_channels==1 and self.dense_init_units==1 :
        #   self.activation = IdentityActivation()
        #else:
        #   self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        self.activation = F.leaky_relu

        self.conv1 = CliffordConv2d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv2d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv2d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv2d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.conv_out_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense4 = nn.Linear(self.num_blades, 1, bias=True)

        #self.avg_pool_1 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_2 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_3 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_4 = nn.AvgPool2d(self.pooling_size)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
    
    def forward(self, inputs):

        #import code; code.interact(local=locals()) 

        B, W, _, H = inputs.size()

        multivectors = torch.zeros(B, W, H, self.num_blades).to(self.device)
        multivectors[:, :, :, self.blades_idxs] = inputs.reshape(B, W, H, 1) #if C>1 else inputs.squeeze()

        B, W, H, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, W, H, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _W, _H, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _W, _H)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _W, _H, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _W, _H, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _W, _H)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _W, _H, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _W, _H, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _W, _H)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _W, _H, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _W, _H, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _W, _H)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        c4 = self.global_avg_pool(c4)
        c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d3 = d3.reshape(B, S_, Q)
        d3 = d3.reshape(B*S_, Q)

        #out = d3[:, :, self.blades_idxs[1]].squeeze()

        out = self.dense4(d3)
        out = out.view(B, S_)

        return out

  
class CliffSER2D_EMD(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffSER2D_EMD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = 0
        #self.pooling_size = model_params['pooling_size']

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.3

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        #if self.conv_hidden_channels==1 and  self.conv_out_channels==1 and self.dense_init_units==1 :
        #   self.activation = IdentityActivation()
        #else:
        #   self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        self.activation = F.leaky_relu

        self.conv1 = CliffordConv2d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv2d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv2d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv2d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.conv_out_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        #self.dense4 = nn.Linear(self.num_blades, 1, bias=True)

        #self.avg_pool_1 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_2 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_3 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_4 = nn.AvgPool2d(self.pooling_size)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
    
    def forward(self, inputs):

        B, _ , W, _, H = inputs.size()
        C = len(self.blades_idxs)

        multivectors = torch.zeros(B, W, H, self.num_blades).to(self.device)
        multivectors[:, :, :, self.blades_idxs[0]] = inputs.reshape(B, W, H, C)[...,3] #if C>1 else inputs.squeeze()
        multivectors[:, :, :, self.blades_idxs[1]] = inputs.reshape(B, W, H, C)[...,0]
        multivectors[:, :, :, self.blades_idxs[2]] = inputs.reshape(B, W, H, C)[...,1]
        multivectors[:, :, :, self.blades_idxs[3]] = inputs.reshape(B, W, H, C)[...,2]

        B, W, H, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, W, H, Q)

        #import code; code.interact(local=locals()) 

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _W, _H, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _W, _H)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _W, _H, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _W, _H, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _W, _H)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _W, _H, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _W, _H, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _W, _H)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _W, _H, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _W, _H, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _W, _H)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        c4 = self.global_avg_pool(c4)
        c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d3 = d3.reshape(B, S_, Q)
        d3 = d3.reshape(B*S_, Q)

        #out = d3[:, :, self.blades_idxs[1]].squeeze()

        out = self.dense4(d3)
        out = out.view(B, S_)

        return out

class PureCliffSER2D(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(PureCliffSER2D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = model_params['kernel_size']//2
        #self.pooling_size = model_params['pooling_size']

        self.dense_init_dim1 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(128))))
        #self.dense_init_dim1 = self.dense_init_dim1//self.pooling_size**2
        self.dense_init_dim2 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(188))))
        #self.dense_init_dim2 = self.dense_init_dim2//self.pooling_size**2
        #self.dense_init_units = model_params['dense_units'][2]
        self.dense_units_in_channels = self.dense_init_dim1 * self.dense_init_dim2 * self.conv_out_channels

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_hidden_channels3 = model_params['dense_units'][6]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.2

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        if self.conv_hidden_channels1==1:
           self.activation = IdentityActivation()
        else:
           self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        #self.activation = F.silu #F.leaky_relu

        self.conv1 = CliffordConv2d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv2 = CliffordConv2d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv3 = CliffordConv2d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv4 = CliffordConv2d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.dense_units_in_channels, self.dense_units_hidden_channels1, bias=True, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=True, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_hidden_channels3, bias=True, sandwich=self.sandwich)
        self.dense4 = CliffordLinear(self.g, self.dense_units_hidden_channels3, self.dense_units_out_channels, bias=True, sandwich=self.sandwich)
        self.dense5 = nn.Linear(self.num_blades, 1, bias=True)

        #self.avg_pool_1 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_2 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_3 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_4 = nn.AvgPool2d(self.pooling_size)

        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
        self.dropout_8 = nn.Dropout(self.dropout_rateDense)
    
    def update_dim(self, S):
        if S%2==0:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride)+1))
        else:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
    
    def forward(self, inputs):

        B, W, _, H = inputs.size()
        C = len(self.blades_idxs)

        multivectors = torch.zeros(B, W, H, self.num_blades).to(self.device)
        multivectors[:, :, :, self.blades_idxs] = inputs.reshape(B, W, H, C) #if C>1 else inputs.squeeze()

        B, W, H, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, W, H, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _W, _H, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _W, _H)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _W, _H, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _W, _H, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _W, _H)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _W, _H, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _W, _H, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _W, _H)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _W, _H, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _W, _H, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _W, _H)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        c4 = c4.reshape(B, C4*_W*_H, Q)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        #c4 = self.global_avg_pool(c4)
        #c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d2 = d2.reshape(B, S_, Q)
        d3 = d3.reshape(B, S_, Q)

        d4 = self.activation(self.dense4(d3))
        B, S_, Q = d4.size()
        d4 = d4.reshape(B, Q, S_)
        d4 = self.dropout_8(d4)
        #d2 = d2.reshape(B, S__, Q)
        d4 = d4.reshape(B*S_, Q)

        #out = d1[:, :, 0].squeeze()

        out = self.dense5(d4)
        out = out.view(B, S_)

        return out

class CliffSER2D_DELTA(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(CliffSER2D_DELTA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = 0
        #self.pooling_size = model_params['pooling_size']

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.2

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        #if self.conv_hidden_channels==1 and  self.conv_out_channels==1 and self.dense_init_units==1 :
        #   self.activation = IdentityActivation()
        #else:
        #   self.activation = CliffordSumVSiLU()
        #self.activation = IdentityActivation()
        self.activation = F.leaky_relu

        self.conv1 = CliffordConv2d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv2 = CliffordConv2d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv3 = CliffordConv2d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        self.conv4 = CliffordConv2d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=False, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.conv_out_channels, self.dense_units_hidden_channels1, bias=False, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=False, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_out_channels, bias=False, sandwich=self.sandwich)
        self.dense4 = nn.Linear(self.num_blades, 1, bias=True)

        #self.avg_pool_1 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_2 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_3 = nn.AvgPool2d(self.pooling_size)
        #self.avg_pool_4 = nn.AvgPool2d(self.pooling_size)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
    
    def forward(self, inputs):

        B, C, W, H = inputs.size()

        multivectors = torch.zeros(B, W, H, self.num_blades).to(self.device)
        multivectors[:, :, :, self.blades_idxs[0]] = inputs.reshape(B, W, H, C)[...,0]
        multivectors[:, :, :, self.blades_idxs[1]] = inputs.reshape(B, W, H, C)[...,1]
        multivectors[:, :, :, self.blades_idxs[2]] = inputs.reshape(B, W, H, C)[...,2]

        B, W, H, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, W, H, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _W, _H, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _W, _H)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _W, _H, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _W, _H, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _W, _H)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _W, _H, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _W, _H, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _W, _H)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _W, _H, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _W, _H, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _W, _H)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        c4 = self.global_avg_pool(c4)
        c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d3 = d3.reshape(B, S_, Q)
        d3 = d3.reshape(B*S_, Q)

        #out = d3[:, :, self.blades_idxs[0]].squeeze()

        out = self.dense4(d3)
        out = out.view(B, S_)

        return out

class PureCliffSER2D_DELTA(nn.Module):

    def __init__(self, model_params):
        # kernel_size=10, stride=1, padding=1, dense_units=[128], g=[-1, -1], blades_idxs = [1]

        super(PureCliffSER2D_DELTA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_in_channels = 1
        self.conv_hidden_channels1 = model_params['dense_units'][0]
        self.conv_hidden_channels2 = model_params['dense_units'][1]
        self.conv_hidden_channels3 = model_params['dense_units'][2]
        self.conv_out_channels = model_params['dense_units'][3]
        self.kernel_size = model_params['kernel_size']
        self.stride = model_params['stride']
        self.padding = model_params['kernel_size']//2

        self.dense_init_dim1 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(128))))
        self.dense_init_dim2 = self.update_dim(self.update_dim(self.update_dim(self.update_dim(188))))
        self.dense_units_in_channels = self.dense_init_dim1 * self.dense_init_dim2 * self.conv_out_channels

        self.dense_units_hidden_channels1 = model_params['dense_units'][4]
        self.dense_units_hidden_channels2 = model_params['dense_units'][5]
        self.dense_units_hidden_channels3 = model_params['dense_units'][6]
        self.dense_units_out_channels = 6 
        self.dropout_rateConv = 0.2
        self.dropout_rateDense = 0.2

        self.g = model_params['g']
        self.num_blades = 2**len(self.g)
        self.blades_idxs = model_params['blades_idxs']
        self.sandwich = False

        if self.conv_hidden_channels1==1:
           self.activation = IdentityActivation()
        else:
           self.activation = CliffordSumVSiLU()

        self.conv1 = CliffordConv2d(self.g, self.conv_in_channels, self.conv_hidden_channels1, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv2 = CliffordConv2d(self.g, self.conv_hidden_channels1, self.conv_hidden_channels2, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv3 = CliffordConv2d(self.g, self.conv_hidden_channels2, self.conv_hidden_channels3, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        self.conv4 = CliffordConv2d(self.g, self.conv_hidden_channels3, self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride, 
                                    padding=self.padding, bias=True, sandwich=self.sandwich)
        
        self.dense1 = CliffordLinear(self.g, self.dense_units_in_channels, self.dense_units_hidden_channels1, bias=True, sandwich=self.sandwich)
        self.dense2 = CliffordLinear(self.g, self.dense_units_hidden_channels1, self.dense_units_hidden_channels2, bias=True, sandwich=self.sandwich)
        self.dense3 = CliffordLinear(self.g, self.dense_units_hidden_channels2, self.dense_units_hidden_channels3, bias=True, sandwich=self.sandwich)
        self.dense4 = CliffordLinear(self.g, self.dense_units_hidden_channels3, self.dense_units_out_channels, bias=True, sandwich=self.sandwich)
        self.dense5 = nn.Linear(self.num_blades, 1, bias=True)
        
        self.dropout_1 = nn.Dropout(self.dropout_rateConv)
        self.dropout_2 = nn.Dropout(self.dropout_rateConv)
        self.dropout_3 = nn.Dropout(self.dropout_rateConv)
        self.dropout_4 = nn.Dropout(self.dropout_rateConv)
        self.dropout_5 = nn.Dropout(self.dropout_rateDense)
        self.dropout_6 = nn.Dropout(self.dropout_rateDense)
        self.dropout_7 = nn.Dropout(self.dropout_rateDense)
        self.dropout_8 = nn.Dropout(self.dropout_rateDense)
    
    def update_dim(self, S):
        if S%2==0:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride)+1))
        else:
            if self.stride%2==0 and self.kernel_size%2==0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2==0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
            elif self.stride%2!=0 and self.kernel_size%2==0:
                return int(np.ceil(((S-self.kernel_size+2*self.padding)/self.stride))+1)
            elif self.stride%2!=0 and self.kernel_size%2!=0:
                return int(np.floor(((S-self.kernel_size+2*self.padding)/self.stride)+1))
    
    def forward(self, inputs):

        B, C, W, H = inputs.size()

        multivectors = torch.zeros(B, W, H, self.num_blades).to(self.device)
        multivectors[:, :, :, self.blades_idxs[0]] = inputs.reshape(B, W, H, C)[...,0]
        multivectors[:, :, :, self.blades_idxs[1]] = inputs.reshape(B, W, H, C)[...,1]
        multivectors[:, :, :, self.blades_idxs[2]] = inputs.reshape(B, W, H, C)[...,2]

        B, W, H, Q = multivectors.size()
        multivectors = multivectors.reshape(B, 1, W, H, Q)

        c1 = self.activation(self.conv1(multivectors))
        B, C1, _W, _H, Q = c1.size()
        c1 = c1.reshape(B, C1*Q, _W, _H)
        #c1 = self.avg_pool_1(c1)
        c1 = self.dropout_1(c1)
        c1 = c1.reshape(B, C1, _W, _H, Q)
        #c1 = c1.reshape(B, C1, _W//self.pooling_size, _H//self.pooling_size, Q)

        c2 = self.activation(self.conv2(c1))
        B, C2, _W, _H, Q = c2.size()
        c2 = c2.reshape(B, C2*Q, _W, _H)
        #c2 = self.avg_pool_2(c2)
        c2 = self.dropout_2(c2)
        c2 = c2.reshape(B, C2, _W, _H, Q)
        #c2 = c2.reshape(B, C2, _W//self.pooling_size, _H//self.pooling_size, Q)

        c3 = self.activation(self.conv3(c2))
        B, C3, _W, _H, Q = c3.size()
        c3 = c3.reshape(B, C3*Q, _W, _H)
        #c3 = self.avg_pool_2(c3)
        c3 = self.dropout_3(c3)
        c3 = c3.reshape(B, C3, _W, _H, Q)
        #c3 = c3.reshape(B, C3, _W//self.pooling_size, _H//self.pooling_size, Q)

        c4 = self.activation(self.conv4(c3))
        B, C4, _W, _H, Q = c4.size()
        c4 = c4.reshape(B, C4*Q, _W, _H)
        #c4 = self.avg_pool_2(c4)
        c4 = self.dropout_4(c4)
        c4 = c4.reshape(B, C4*_W*_H, Q)
        #c4 = c4.reshape(B, C4*Q, _W//self.pooling_size, _H//self.pooling_size)

        #c4 = self.global_avg_pool(c4)
        #c4 = c4.reshape(B, C4, Q)
        
        d1 = self.activation(self.dense1(c4))
        B, S_, Q = d1.size()
        d1 = d1.reshape(B, Q, S_)
        d1 = self.dropout_5(d1)
        d1 = d1.reshape(B, S_, Q)

        d2 = self.activation(self.dense2(d1))
        B, S_, Q = d2.size()
        d2 = d2.reshape(B, Q, S_)
        d2 = self.dropout_6(d2)
        #d2 = d2.reshape(B, S_, Q)
        d2 = d2.reshape(B, S_, Q)

        d3 = self.activation(self.dense3(d2))
        B, S_, Q = d3.size()
        d3 = d3.reshape(B, Q, S_)
        d3 = self.dropout_7(d3)
        #d2 = d2.reshape(B, S_, Q)
        d3 = d3.reshape(B, S_, Q)

        d4 = self.activation(self.dense4(d3))
        B, S_, Q = d4.size()
        d4 = d4.reshape(B, Q, S_)
        d4 = self.dropout_8(d4)
        #d2 = d2.reshape(B, S__, Q)
        d4 = d4.reshape(B*S_, Q)

        #out = d1[:, :, 0].squeeze()

        out = self.dense5(d4)
        out = out.view(B, S_)

        return out


class Trainer():

    def __init__(self, model_class, param_dict, loss_type, max_epochs, batch_size, model_dir, visualize=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.min_valid_loss = None
        self.min_valid_acc = None
        self.min_train_loss = None
        self.min_train_acc = None
        self.best_epoch = 0
        self.model_dir = model_dir
        self.param_dict = param_dict
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.patience = 10 #after patience epochs of worse performaces, STOP
        self.tolerance = 1e-5 #tollerance over the validation loss 
        self.n_epochs_no_improvement = 0 #accumulator
        self.visualize = visualize
        self.model = model_class(self.param_dict).to(self.device)
        self.best_wights = None
        self.loss_type = loss_type
        if self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropy() 
        print("Model is on device:", self.device)
    
    def train(self, train_loader, val_loader, lr, weight_decay):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=self.max_epochs, steps_per_epoch=len(train_loader))
        criterion = nn.CrossEntropyLoss()

        self.model.train() 

        plot_train_loss = []
        plot_train_acc = []
        plot_val_loss = []
        plot_val_acc = []
        lrs = []
        
        for epoch in range(self.max_epochs):
            print(f"====Training Epoch: {epoch}====")
            
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            tbar = tqdm(train_loader, ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
        
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()

                outputs = self.model(data)
                
                loss = criterion(outputs, labels)
                _, predicted_class = torch.max(outputs.data, 1)
                _, true_class = torch.max(labels, 1)
                total = true_class.size(0)
                correct = (predicted_class == true_class).sum().item()

                loss.backward()

                optimizer.step()
                scheduler.step()

                running_total += total
                running_correct += correct
                running_loss += loss.item()

                tbar.set_postfix({"loss": loss.item(), "acc": correct/total, "lr": optimizer.param_groups[0]["lr"]})

            train_loss = running_loss / len(train_loader)
            train_acc = running_correct / running_total
            val_loss, val_acc = self.validate(val_loader)

            plot_train_loss.append(train_loss)
            plot_train_acc.append(train_acc)
            plot_val_loss.append(val_loss)
            plot_val_acc.append(val_acc)     

            print(f"Epoch {epoch}: training error = {train_loss:.4f} (accuracy = {train_acc:.4f}), validation error = {val_loss:.4f} (accuracy = {val_acc:.4f})")

            if self.min_valid_loss is None or val_loss < self.min_valid_loss - self.tolerance:
                self.min_valid_loss = val_loss
                self.min_valid_acc = val_acc
                self.min_train_loss = train_loss
                self.min_train_acc = train_acc
                self.best_epoch = epoch
                self.best_weights = self.model.state_dict()
                self.n_epochs_no_improvement = 0
                print("Update best model! Best epoch: {}".format(self.best_epoch))
            else:
                self.n_epochs_no_improvement += 1
            
            
            if self.n_epochs_no_improvement >= self.patience:
                print(f"Early stopping after epoch {epoch}\n")
                self.model.load_state_dict(self.best_weights)
                break
        
        print(f"Best trained epoch = {self.best_epoch:.4f}\n")
        print(f"Training error: {self.min_train_loss:.4f} (accuracy: {self.min_train_acc:.4f})\n")
        print(f"Validation error: {self.min_valid_loss:.4f} (accuracy: {self.min_valid_acc:.4f})")
        self.save_model('Best_epoch', '')
        np.save(self.model_dir+'/Best_epoch_idx', self.best_epoch)

        if self.visualize:
            self.plot_learning_epochs(plot_train_loss, plot_val_loss, plot_train_acc, plot_val_acc, lrs)
        
        print('Finished Training')
        
        return self.min_train_loss, self.min_train_acc

    def validate(self, val_loader):
        print(f"===Validating===")
        
        val_loss = 0.0
        correct = 0
        total = 0

        self.model.eval()
        
        with torch.no_grad():
            vbar = tqdm(val_loader, ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted_class = torch.max(outputs.data, 1)
                _, true_class = torch.max(labels, 1)
                total += true_class.size(0)
                correct += (predicted_class == true_class).sum().item()

                vbar.set_postfix(loss=loss.item())
        
        accuracy = correct / total
        return val_loss / len(val_loader), accuracy
    
    def save_model(self, name, index):
        model_path = os.path.join(
            self.model_dir, name + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
    
    def plot_learning_epochs(self, mean_training_losses, mean_valid_losses, mean_training_acc, mean_valid_acc, lrs):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(mean_training_losses)), mean_training_losses, label='Train Loss')
        plt.plot(range(len(mean_valid_losses)), mean_valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses')
        plt.legend()
        plt.savefig(self.model_dir+'learning_losses.pdf', bbox_inches='tight')

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(mean_training_acc)), mean_training_acc, label='Train Accuracy')
        plt.plot(range(len(mean_valid_acc)), mean_valid_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracy')
        plt.legend()
        plt.savefig(self.model_dir+'learning_accuracy.pdf', bbox_inches='tight')

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.savefig(self.model_dir+'learning_rates.pdf', bbox_inches='tight')


# Processor class

class Processor():
    
    def __init__(self, model_class, param_dict, loss_type, load_path, type='pretrain', use_last_epoch=False):
        self.load_path = load_path
        self.type = type                        
        self.loss_type = loss_type
        self.use_last_epoch=use_last_epoch
        self.param_dict = param_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.model = model_class(self.param_dict).to(self.device)
        if self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropy() 
        if type=='pretrain':
            if not os.path.exists(self.load_path):
                raise ValueError("Pre-trained model path error! ")
            self.model.load_state_dict(torch.load(self.load_path))
            print("Testing using specified pretrained model!")
        else:
            if self.use_last_epoch:
                last_epoch_model_path = os.path.join(self.load_path, 'Epoch_' + str(self.get_last_epoch()) + '.pth')
                print("Testing using the last epoch!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(self.load_path, 'Best_epoch.pth')
                print("Testing using the best epoch!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

    def predict(self, data_loader):       
        print('')
        print("===Testing===")

        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
        
                data = batch[0].to(self.device)
                preds = self.model(data)

                predictions.append(preds.cpu())
                
        return np.concatenate(predictions)
    
    def score(self, outputs, labels):
        outputs = torch.FloatTensor(outputs)
        labels = torch.FloatTensor(labels)

        loss = self.criterion(outputs, labels)
        _, predicted_class = torch.max(outputs, 1)
        _, true_class = torch.max(labels, 1)
        total = labels.size(0)
        correct = (predicted_class == true_class).sum().item()

        test_loss = loss.item()
        test_acc = correct / total

        return test_loss, test_acc 
    
    def get_last_epoch(self):
        pth_files = [f for f in os.listdir(self.load_path) if f.endswith('.pth') and 'Epoch_' in f]
        return max([int(f.split('Epoch_')[1].split('.pth')[0]) for f in pth_files])
