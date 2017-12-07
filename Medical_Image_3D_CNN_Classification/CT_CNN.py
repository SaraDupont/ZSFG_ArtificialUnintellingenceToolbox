#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:32:55 2017

@author: davidbmccoy
"""
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
import 3d_CNN_CT_Classification
import dcmstack
from glob import glob


def get_parser_data():
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
    parser.add_argument("-proc_folder",
                        help="Name of the procedure folder you're doing the classification on. For ex 2_0mm_Axial_Brain_Std\".",
                        type=str,
                        dest="proc_folder",
                        default="")
    
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
    return parser

def get_parser():
    parser_data = get_parser_data()
    #
    parser = argparse.ArgumentParser(description="Classification function based on 3D convolutional neural networks", parents=[parser_data])
    parser.add_argument("-split",
                        help="Split ratio between train and test sets. Values should be between 0 and 1. Example: -split 0.4 would use 40 percent of the data for training and 60 percent for testing.",
                        type=restricted_float,
                        dest="split",
                        default=0.8)
    parser.add_argument("-valid-split",
                        help="Split ratio between validation and actual train sets within the training set. Values should be between 0 and 1. Example: -split 0.3 would use 30 percent of the training set for validation (within the model training) and 70 percent for actual training.",
                        type=restricted_float,
                        dest="valid_split",
                        default=0.2)
    parser.add_argument("-num-layer",
                        help="Number of layers in the  contracting path of the model",
                        type=int,
                        dest="num_layer",
                        default=4)
    parser.add_argument("-imsize",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size",
                        default=512)
    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=5000)
    return parser

class Hemorrhage_Classification():
    
    def __init__(self, param):
        self.param = param # parser parameters

        self.list_subjects = [] # list of subjects (objects of type Subject)

        self.list_orientation = [] # list of original orientation of images = to be able to put back the results into the original image orientation at the end of the segmentaiton
        self.list_im = [] # list of original images data
        self.list_mask = []  # list of original masks data
        self.list_headers= [] # list of headers
        #
        self.list_subj_train = [] # list of subjects used to train model
        self.list_subj_valid = [] # list of subjects used to validate model within training
        self.list_subj_test = [] # list of subjects used to test model
        #
        self.train_imgs_tensor = [] # list of preprocessed images as ndarrays to train model on
        self.train_masks_tensor = []  # list of preprocessed masks as ndarrays to test model on
        self.valid_imgs_tensor = [] # list of preprocessed images as ndarrays to validate model on
        self.valid_masks_tensor = [] # list of preprocessed masks as ndarrays to validate model on
        #
        self.smooth_dc = 1.
        self.width = 256
        self.height = 256
        self.depth = 40
        self.nLabel = 2
        
        #
        self.model_train_hist = None

        K.set_image_data_format('channels_last')  # defined as b/w images throughout
        sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
        self.y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

        
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    ## creates a tensor with shape = shape of constant values = 0.1
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
    ## Convolution and Pooling
    # Convolution here: stride=1, zero-padded -> output size = input size
    def conv3d(x, W):
      return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]
    ## not sure here why there is five dimensions in stride...
    
    # Pooling: max pooling over 2x2 blocks
    def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
      return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')
    ## it looks here that the kernel size or block size is 4x4x4
    
    ## First Convolutional Layer
    # Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
    W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
    b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]
    
    # Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
    x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
    print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1
    
    # x_image * weight tensor + bias -> apply ReLU -> apply max-pool
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
    print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
    h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 
    print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32
    
    
    ## Second Convolutional Layer
    # Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
    W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
    b_conv2 = bias_variable([64]) # [64]
    
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
    print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
    h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
    print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64
    
    
    ## Densely Connected Layer (or fully-connected layer)
    # fully-connected layer with 1024 neurons to process on the entire image
    W_fc1 = weight_variable([16*16*3*64, 1024])  # [7*7*64, 1024]
    print(W_fc1.shape)
    b_fc1 = bias_variable([1024]) # [1024]]
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*3*64])  # -> output image: [-1, 7*7*64] = 3136
    print(h_pool2_flat.get_shape)  # (?, 2621440)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
    print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024
    
    ## Dropout (to reduce overfitting; useful when training very large neural network)
    # We will turn on dropout during training & turn off during testing
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print(h_fc1_drop.get_shape)  # -> output: 1024
    
    ## Readout Layer
    W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
    b_fc2 = bias_variable([nLabel]) # [10]
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print(y_conv.get_shape)  # -> output: 10
    
    ## Train and Evaluate the Model
    # set up for optimization (optimizer:ADAM)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    # Include keep_prob in feed_dict to control dropout rate.
    for i in range(100):
        batch = get_data_CT(sess,'Train',20)
        # Logging every 100th iteration in the training process.
        if i%5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    # Evaulate our accuracy on the test data
    testset = get_data_CT(sess,'Test',30)
    print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y_: teseset[1], keep_prob: 1.0}))
    
    
    def create_nifti(param):
    list_subjects = []
    for group in os.listdir(param.path):
        if os.path.isdir(os.path.join(param.path, group)):
            for batch in os.listdir(os.path.join(param.path, group)):
                dicom_sorted_path  = os.path.join(param.path, group, batch, 'DICOM-SORTED')
                if os.path.isdir(os.path.join(dicom_sorted_path)):
                    for subj in os.listdir(os.path.join(dicom_sorted_path, subj)):
                        if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                            for contrast in os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                if param.contrast_path in contrast:
                                    for proc in os.path.isdir(os.path.join(dicom_sorted_path, subj, contrast)):
                                        if param.procedure_path in proc:
                                            path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
                                            for fname in os.listdir(path_study):
                                                if fname.endswith('.nii.gz'):
                                                    break
                                            else:
                                                src_dcms = glob(path_study+'*.dcm')
                                                stacks = dcmstack.parse_and_stack(src_dcms)
                                                stack = stacks.values[0]
                                                nii = stack.to_nifti()
                                                nii.to_filename(subj+contrast+proc+'.nii.gz')
                                                list_subjects.append(Subject(path=path_study, fname_im=subj+contrast+proc+'.nii.gz'))
                                                
    return list_subjects
                                            
                                            
                                            
                                            
                                        
                                        

                                            