import os
import argparse
from utils import *
import nibabel as nib
import tensorflow as tf
from sklearn.cross_validation import train_test_split

def get_parser():
    parser = argparse.ArgumentParser(description="Segmentation function based on 2D convolutional neural networks")
    parser.add_argument("-data",
                        help="Data to train and/or test the segmentation on",
                        type=str,
                        dest="path")
    parser.add_argument("-c",
                        help="Name of the contrast folder you're doing the segmentation on. For ex \"T2\" or \"T2_ax\".",
                        type=str,
                        dest="contrast",
                        default="T2")
    parser.add_argument("-im",
                        help="String to look for in the images name.",
                        type=str,
                        dest="im",
                        default="im")
    parser.add_argument("-mask",
                        help="String to look for in the masks name.",
                        type=str,
                        dest="mask",
                        default="mask")
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
        self.param = param # parser parameters
        self.list_subj = [] # list of paths to each subject
        self.list_fname_im = [] # list of images file names
        self.list_fname_mask = [] # list of masks file names
        self.list_orientation = [] # list of original orientation of images = to be able to put back the results into the original image orientation at the end of the segmentaiton
        self.list_im = [] # list of nib images
        self.list_mask = []  # list of nib masks
        self.list_subj_train = [] # list of subjects used to train model
        self.list_subj_test = [] # list of subjects used to test model
        self.list_im_train = [] # list of preprocessed images as ndarrays to train model on
        self.list_im_test = []  # list of preprocessed images as ndarrays to test model on
        self.list_mask_train= []  # list of preprocessed masks as ndarrays to train model on
        self.list_mask_test = []  # list of preprocessed masks as ndarrays to test model on

    def preprocessing(self):
        # get data
        for subj in os.listdir(self.param.path):
            path_contrast = os.path.join(self.param.path, subj, self.param.contrast)
            found_im = False
            found_mask = False
            for f in os.listdir(path_contrast):
                if self.param.im in f and not found_im:
                    self.list_fname_im.append(f)
                    found_im = True
                elif self.param.mask in f and not found_mask:
                    self.list_fname_mask.append(f)
                    found_mask = True
            if found_im and found_mask:
                self.list_subj.append(path_contrast)
        #
        # assert is used to check if a statement is true, if not, it raises an error and stops the program
        assert len(self.list_fname_im) == len(self.list_fname_mask), "ERROR: not the same number of images and masks"
        assert len(self.list_subj) == len(self.list_fname_im), "ERROR not the same number of images and subjects"
        #
        list_im_preprocessed = []
        list_mask_preprocessed = []
        for i, path_subj in enumerate(self.list_subj):
            # load images
            self.list_im.append(nib.load(os.path.join(path_subj, self.list_fname_im[i])))
            self.list_mask.append(nib.load(os.path.join(path_subj, self.list_fname_mask[i])))
            #
            # change orientation to make sure axial slices are in the correct dimension (= axial in the third dimension)
            im_ori, im_reorient = self.reorient(self.list_im[i])
            mask_ori, mask_reorient = self.reorient(self.list_mask[i])
            assert im_ori == mask_ori, "ERROR: image and mask don't have the same orientation"
            self.list_orientation.append(im_ori)
            #
            # resample/interpolate to param.im_size with tensorflow -> force interpolation to a square image
            im_resample = self.resample(im_reorient)
            mask_resample = self.resample(mask_reorient)
            #
            # TODO: IF OTHER PREPROCESSING STEPS ON ALL IMAGES ARE NEEDED, ADD HERE
            #
            # add preprocessed images and masks to list
            list_im_preprocessed.append(im_resample)
            list_mask_preprocessed.append(mask_resample)
        #
        # select subjects used for training and testing
        self.list_subj_train, self.list_subj_test, list_im_train, list_im_test, list_mask_train, list_mask_test = train_test_split(self.list_subj, list_im_preprocessed, list_mask_preprocessed, test_size=1-self.param.split, train_size=self.param.split)
        self.list_im_train = [im.data for im in list_im_train]
        self.list_im_test = [im.data for im in list_im_test]
        self.list_mask_train = [mask.data for mask in list_mask_train]
        self.list_mask_test = [mask.data for mask in list_mask_test]
        self.list_mask_test = [mask.data for mask in list_mask_test]

    def reorient(self, image):
        # change the orientation of an image
        pass
        # TODO: CHANGE IMAGE ORIENTATION:
        # TODO --> GET CURRENT ORIENTATION AND STORE IT
        # TODO --> CHANGE ORDER OF AXIS, MAYBE HAVE AS A PARAMETER WHAT IS THE ORIENTATION OF SLICES WE WANT TO SEGMENT ?? (I.E. AXIAL, CORONAL OR SAGITAL)
        ori = None #... # get image orientation
        image_reorient = None #...
        #
        return ori, image_reorient

    def resample(self, image):
        # resample an image to a square of size param.im_size
        image_resample = tf.resample(image, size=self.param.im_size) # TODO: CHANGE SYNTAX HERE
        return image_resample


def main():
    parser = get_parser()
    param = parser.parse_args()
    print "here"
    seg = Segmentation(param=param)
    seg.preprocessing()



if __name__=="__main__":
    main()