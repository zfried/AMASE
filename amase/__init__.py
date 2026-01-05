"""
AMASE - Mixture Analysis Algorithm for rotational spectroscopy
"""

# Import compatibility shim for old pickle files
import sys
from . import molsim
sys.modules['molsim'] = molsim

from .main import run_assignment

__version__ = "0.1.0"

__all__ = ['run_assignment']
