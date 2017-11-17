#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:33:55 2017

@author: davidbmccoy
"""

# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
import dcmstack
from glob import glob

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 256
FLAGS.height = 256
FLAGS.depth = 40 # 3
batch_index = 0
filenames = []

# user selection
FLAGS.data_dir = '/media/mccoyd2/hamburger/CT_Hemorrhage'
FLAGS.num_class = 2

## data_set is equal to training or test set
## text file with labels
## files are in data, under training/test, and named label
def get_filenames(data_set):
    global filenames
    labels = []

    with open(FLAGS.data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list
## os.listdir will list all the files in the path given, therefore 
## training -> hemorrhage cases -> list the file names for these cases
## same for training -> controls, and the testset
    for i, label in enumerate(labels):
        list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label)
        for filename in list:
            filenames.append([label + '/' + filename, i])

    random.shuffle(filenames)

## so this get_filenames function creates the filenames for each set
## and randomly shuffles them

def get_data_CT(sess, data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_set) 
    max = len(filenames) ## total number of data for each session

    begin = batch_index
    end = batch_index + batch_size 

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, FLAGS.num_class)) # zero-filled list for 'one hot encoding'
    index = 0

    for i in range(begin, end):
        
        imagePath = FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0]
        
        check_nifti = glob('*.nii.gz') 
        
        if check_nifti == []:
            src_dcms = glob('*.dcm')
            stacks = dcmstack.parse_and_stack(src_dcms)
            stack = stacks.values[0]
            nii = stack.to_nifti()
            nii.to_filename(filenames[i][0]+'.nii.gz')
        else: 
            pass
        
        CT_org = nib.load(imagePath)
        CT_data = CT_org.get_data()  # 256x256x40; numpy.ndarray
        
        # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
        resized_image = tf.image.resize_images(images=CT_data, size=(FLAGS.width,FLAGS.height), method=1)
        # may need to resize the z-dimension here
        image = sess.run(resized_image)  # (256,256,40)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data[index][filenames[i][1]] = 1  # assign 1 to corresponding column (one hot encoding)
        index += 1

    batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)

    return x_data_, y_data