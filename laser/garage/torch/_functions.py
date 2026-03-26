"""Utility functions for PyTorch algorithms.

A collection of common functions that are used by Pytorch algos.

This collection of functions can be used to manage the following:
    - Pytorch GPU usage
        - setting the default Pytorch GPU
        - converting Tensors to GPU Tensors
        - Converting Tensors into `numpy.ndarray` format and vice versa
    - Updating model parameters
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


def set_gpu_mode(mode, gpu_id=0):
    """Set GPU mode and device ID.

    Args:
        mode (bool): Whether or not to use GPU
        gpu_id (int): GPU ID

    """
    # pylint: disable=global-statement
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    gpu_name = 'cuda'
    if gpu_id is not None:
        gpu_name += ':' + str(_GPU_ID)
    _DEVICE = torch.device(gpu_name if _USE_GPU else 'cpu')


def global_device():
    """Returns the global device that torch.Tensors should be placed on.

    Note: The global device is set by using the function
        `garage.torch._functions.set_gpu_mode.`
        If this functions is never called
        `garage.torch._functions.device()` returns None.

    Returns:
        `torch.Device`: The global device that newly created torch.Tensors
            should be placed on.

    """
    # pylint: disable=global-statement
    global _DEVICE
    return _DEVICE


# pylint: disable=W0223
class NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        elif isinstance(non_linear, str):
            self.module = getattr(F, non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        """object representation method."""
        return repr(self.module)
