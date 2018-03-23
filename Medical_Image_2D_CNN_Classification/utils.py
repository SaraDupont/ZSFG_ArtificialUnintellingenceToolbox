import argparse
import errno
import os
from glob import glob

def restricted_float(x=None):
    if x is not None:
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def add_suffix(path, suffix):
    list_path = path.split('/')
    fname = list_path[-1]
    list_fname = fname.split('.')
    list_fname[0]+= suffix
    fname_suffix = '.'.join(list_fname)
    list_path[-1] = fname_suffix
    path_suffix = '/'.join(list_path)
    return path_suffix

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
