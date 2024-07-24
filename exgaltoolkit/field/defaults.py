import logging

ID      = "FieldDefaultID"
N       = 768
Lbox    = 7700.
Nside   = 1024
nlpt    = 2
input   = "lptfiles"
gpu     = True
mpi     = True
loglev  = logging.ERROR # xgfield logging uses standard levels from logging module
is64bit = False
peak_per_cell_memory_in_MB = 100.
cwsp    = None
