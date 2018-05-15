"""
Takes in an ENDF file that has a covariance on it then creates NJOY
inputs to generate group covariances in BOXER format. This format is
converted to matrix format then parsed into a ASAPy covariance store.
The result can be appended to an existing store or made into a new store.
"""

from ASAPy import asapy_covformat
from ASAPy import njoy