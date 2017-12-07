#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:13:35 2017

@author: mccoyd2
"""
from utils import *
import os, glob
import argparse
from utils import *
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
from msct_image import Image
from keras.models import load_model
from keras.callbacks import History 
import pandas as pd 
from keras.utils import plot_model
from scipy import stats
import matplotlib.pyplot as plt
from skimage.transform import resize
import input_3Dimage
import tensorflow as tf
#import 3d_CNN_CT_Classification
import dicom2nifti
from glob import glob
import commands
import dcmstack

def get_parser_classify():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data",
                        help="Data to train and/or test the classification on",
                        type=str,
                        dest="path")
    parser.add_argument("-cfolder",
                        help="Name of the contrast folder you're doing the classification on. For ex CT_BRAIN_WO_CONTRAST.",
                        type=str,
                        dest="contrast_folder",
                        default="")
    parser.add_argument("-slice_thickness",
                        help="Slice thickness of the study",
                        type=str,
                        dest="slice_thickness",
                        default="")
    parser.add_argument("-direction",
                        help="direction of the study",
                        type=str,
                        dest="direction",
                        default="Axial")
    
    parser.add_argument("-im",
                        help="String to look for in the images name.",
                        type=str,
                        dest="im",
                        default="im")

    
    return parser

class Subject():
    def __init__(self, path='', group='', ori=''):
        self.path = path
        self.group = group

    #
    def __repr__(self):
        to_print = '\nSubject:   '
        to_print += '   path: '+self.path
        to_print += '   group: '+self.group

        return to_print
    
    
class Hemorrhage_Classification():
    
    def __init__(self, param):
        self.param = param # parser parameters

    def create_nifti(self):
        list_subjects = []
        failed_nifti_conv_subjects = [] 
        for group in os.listdir(self.param.path):
            if os.path.isdir(os.path.join(self.param.path, group)):
                for batch in os.listdir(os.path.join(self.param.path, group)):
                    dicom_sorted_path  = os.path.join(self.param.path, group, batch, 'DICOM-SORTED')
                    if os.path.isdir(os.path.join(dicom_sorted_path)):
                        for subj in os.listdir(os.path.join(dicom_sorted_path)):
                            if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                for contrast in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                    if self.param.contrast_folder in contrast:
                                        for proc in os.listdir(os.path.join(dicom_sorted_path, subj, contrast)):
                                            if re.findall(r"2.*mm",proc):
                                                if self.param.direction.lower() in proc.lower():
                                                    path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
                                                    for fname in os.listdir(path_study):
                                                        if fname.endswith('.nii.gz'):
                                                            nifti_name = fname
                                                            list_subjects.append(Subject(path=path_study+'/'+nifti_name, group = group))
                                                            break
                                                        
                                                    else:
                                                        #dicom2nifti.dicom_series_to_nifti(path_study, path_study, reorient_nifti=True)
                                                        print("Converting DICOMS for "+subj+" to NIFTI format")
                                                        status, output = commands.getstatusoutput('dcm2nii '+path_study)
                                                        if status != 0:
                                                            failed_nifti_conv_subjects.append(subj)
                                                        else:
                                                            index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                            index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                            nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
                                                            #src_dcms = glob(path_study+'/*.dcm')
        #                                                    stacks = dcmstack.parse_and_stack(src_dcms)
        #                                                    stack = stacks.values[0]
        #                                                    nii = stack.to_nifti()
        #                                                    nii.to_filename(subj+contrast+proc+'.nii.gz')
                                                            list_subjects.append(Subject(path=path_study+'/'+nifti_name,group = group))
        list_subjects = pd.DataFrame(list_subjects)
        list_subjects.to_csv("list_subjects_test.csv")                                            
        return list_subjects

def main():
    parser = get_parser_classify()
    param = parser.parse_args()
    classify = Hemorrhage_Classification(param=param)
    classify.create_nifti()

    ##
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()

