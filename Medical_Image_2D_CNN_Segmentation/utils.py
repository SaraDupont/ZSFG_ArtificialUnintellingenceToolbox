import argparse


def restricted_float(x=None):
    if x is not None:
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x
