import sys
import argparse
from utils import *

def get_parser():
    parser = argparse.ArgumentParser(description="Segmentation function based on 2D convolutional neural networks")
    parser.add_argument("-data",
                        help="Data to train and/or test the segmentation on",
                        type=str,
                        dest="path")
    parser.add_argument("-split",
                        help="Split ratio between train and test sets. Values should be between 0 and 1. Example: -split 0.4 would use 40 percent of the data for training and 60 percent for testing.",
                        type=restricted_float,
                        dest="split",
                        default=0.8)
    parser.add_argument("-imsize",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size",
                        default=128)
    return parser


class Segmentation():
    def __init__(self, param):
        self.param = param


    def preprocessing(self):
        # resample/interpolate to param.im_size with tensorflow
        # tf.resample(size=self.param.im_size)
        pass

def main():
    parser = get_parser()
    param = parser.parse_args()
    print "here"
    seg = Segmentation(param=param)
    seg.preprocessing()



if __name__=="__main__":
    main()