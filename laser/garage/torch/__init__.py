"""PyTorch-backed modules and algorithms."""
# yapf: disable
from laser.garage.torch._functions import (global_device, NonLinearity, set_gpu_mode)

# yapf: enable
__all__ = [
    'global_device', 'set_gpu_mode', 'NonLinearity',
]
