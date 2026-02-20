# __init__.py

"""
pyPRT - python code for solving polarized radiative transfer
"""

__version__ = '0.1.0'
__author__ = 'Guoyin Chen'
__email__ = 'gychen@smail.nju.edu.cn'
__license__ = 'MIT'

from .synth2 import synthesis2 as synth
from .me_atoms import synth_me
from .atoms import atomic_properties
from .pressure import ie_pressure
from .partition_function import u123
from . import synth as synth_old

__all__ = [
    'synth',
    'synth_me',
    'atomic_properties',
    'ie_pressure',
    'u123',
    'synth_old'
]