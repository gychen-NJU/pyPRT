# network/__init__.py
"""
Basic network modules.
"""
from .funcnet import FCN,FuncNet,UserDataset
from .fno import FixedFNOHydrostatic as FNOHydrostatic
from .deeponet import HSE_DeepONet
HSE_FNO = FNOHydrostatic

__all__ = [
    # funcnet
    "FCN","FuncNet","UserDataset",
    # fno
    "FNOHydrostatic","HSE_FNO",
    # deeponet
    "HSE_DeepONet"
]