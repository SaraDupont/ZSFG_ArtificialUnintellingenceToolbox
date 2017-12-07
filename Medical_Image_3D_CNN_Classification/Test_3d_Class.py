#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:13:35 2017

@author: mccoyd2
"""
from utils import *
import os, glob, re
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
from sklearn.preprocessing import LabelBinarizer

FLAGS = tf.app.flags.FLAGS
FLAGS.num_class = 2
FLAGS.data_dir = '/media/mccoyd2/hamburger/Hemorrhage_Study'


def get_parser_classify():
    # classification parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data",
                        help="Data to train and/or test the classification on",
                        type=str,
                        dest="path")
    parser.add_argument("-imaging_plane",
                        help="Axial, Saggital, Coronal etc.",
                        type=str,
                        dest="imaging_plane",
                        default="")
    parser.add_argument("-slice_thickness",
                        help="Slice thickness of the study",
                        type=str,
                        dest="slice_thickness",
                        default="")
    parser.add_argument('--nargs',
                        help="list of relevant study procedures",
                        dest="procedure",
                        default="CT_BRAIN_WO_CONTRAST",
                        nargs='+')
    
    return parser

def get_parser():
    parser_data = get_parser_classify()
    #
    parser = argparse.ArgumentParser(description="Classification function based on 3D convolutional neural networks", parents=[parser_data])
    parser.add_argument("-split",
                        help="Split ratio between train and test sets. Values should be between 0 and 1. Example: -split 0.4 would use 40 percent of the data for training and 60 percent for testing.",
                        type=restricted_float,
                        dest="split",
                        default=0.8)
    parser.add_argument("-valid_split",
                        help="Split ratio between validation and actual train sets within the training set. Values should be between 0 and 1. Example: -split 0.3 would use 30 percent of the training set for validation (within the model training) and 70 percent for actual training.",
                        type=restricted_float,
                        dest="valid_split",
                        default=0.2)
    parser.add_argument("-num_layer",
                        help="Number of layers in the  contracting path of the model",
                        type=int,
                        dest="num_layer",
                        default=4)
    parser.add_argument("-im_size",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size",
                        default=512)
    parser.add_argument("-im_depth",
                    help="Depth of the image used in the CNN.",
                    type=int,
                    dest="im_depth",
                    default=80)
    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=5000)
    parser.add_argument("-batch_size",
                    help="Size of batches that make up each epoch.",
                    type=int,
                    dest="batch_size",
                    default=32)
    
    parser.add_argument("-nlabel",
                    help="Number of disease labels that correspond to images.",
                    type=int,
                    dest="nlabel",
                    default=2)
    
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
        self.param = param 
        self.list_subjects = pd.DataFrame([])
        self.failed_nifti_conv_subjects = []
        self.batch_index = 0        
        self.list_training_subjects = []
        self.list_training_subjects_labels = []
        self.list_test_subjects = []
        self.list_test_subjects_labels = []
        K.set_image_data_format('channels_last')  # defined as b/w images throughout
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, self.param.im_size*self.param.im_size*self.param.im_depth]) # [None, 28*28]
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.param.nlabel])  # [None, 10]

        # Include keep_prob in feed_dict to control dropout rate.
    def run_model(self):
        for i in range(self.param.epochs):
            batch_train = Hemorrhage_Classification.get_CT_data(self.list_subj_train, self.list_subj_train_labels)
            batch_validation = Hemorrhage_Classification.get_CT_data(self.list_subj_valid, self.list_subj_valid_labels)
            # Logging every 100th iteration in the training process.
            if i%100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                valid_accuracy = self.accuracy.eval(feed_dict={self.x:batch_validation[0], self.y_: batch_validation[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g, validation accuracy %g"%(i, train_accuracy,valid_accuracy))
            self.train_step.run(feed_dict={self.x: batch_train[0], self.y_: batch_train[1], self.keep_prob: 0.5})
        
        # Evaulate our accuracy on the test data
        for i in len(self.list_subj_test)/(self.batch_size):
            testset = Hemorrhage_Classification.get_CT_data(self.list_subj_test, self.list_subj_test_labels)
            print("test accuracy %g"%self.accuracy.eval(feed_dict={self.x: self.list_subj_test[0], self.y_: self.list_subj_test_labels, self.keep_prob: 1.0}))

        
    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    ## creates a tensor with shape = shape of constant values = 0.1
    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
    ## Convolution and Pooling
    # Convolution here: stride=1, zero-padded -> output size = input size
    def conv3d(self,x, W):
      return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]
    ## not sure here why there is five dimensions in stride...
    
    # Pooling: max pooling over 2x2 blocks
    def max_pool_2x2(self,x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
      return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    
    def build_vol_classifier(self):
        
        self.features = 32
        self.channels = 1 
        self.list_weight_tensors = [] 
        self.list_bias_tensors = []
        self.list_relu_tensors = []
        self.max_pooled_tensors = []
        self.list_features = []
        self.list_channels = [] 
        
        input_image = tf.reshape(self.x, [-1,self.param.im_size,self.param.im_size,self.param.im_depth,1]) 
        #input_image = tf.reshape(x, [-1,512,512,80,1]) 

        for i in range(self.param.num_layer):
            
            self.list_features.append(self.features)
            self.list_channels.append(self.channels)
            
            print(input_image.get_shape)
            W_conv = Hemorrhage_Classification.weight_variable([5, 5, 5, self.channels, self.features])  
            self.list_weight_tensors.append(W_conv)
            b_conv = Hemorrhage_Classification.bias_variable([self.features])
            self.list_bias_tensors.append(b_conv)
            h_conv = tf.nn.relu(Hemorrhage_Classification.conv3d(input_image, W_conv) + b_conv)  # conv2d, ReLU(x_image * weight + bias)
            self.list_relu_tensors.append(h_conv)
            print(h_conv.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
            input_image = Hemorrhage_Classification.max_pool_2x2(h_conv)  # apply max-pool 
            self.list_max_pooled_tensors.append(input_image)
            print(input_image.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32
            
            if i == 0: 
                self.channels += self.features - 1
                print self.channels
                self.features += self.features
                print self.features
            else: 
                self.channels *= 2 
                print self.channels
                self.features *= 2 
                print self.features
            
            
        ## Densely Connected Layer (or fully-connected layer)
        # fully-connected layer with 1024 neurons to process on the entire image
        W_fc1 = Hemorrhage_Classification.weight_variable([(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features, 1024])  # [7*7*64, 1024]
        print(W_fc1.shape)
        b_fc1 = Hemorrhage_Classification.bias_variable([1024]) # [1024]]
        
        h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, (self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features])  # -> output image: [-1, 7*7*64] = 3136
        print(h_pool2_flat.get_shape)  # (?, 2621440)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
        print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024
        
        ## Dropout (to reduce overfitting; useful when training very large neural network)
        # We will turn on dropout during training & turn off during testing
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1_drop.get_shape)  # -> output: 1024
        
        ## Readout Layer
        W_fc2 = Hemorrhage_Classification.weight_variable([1024, self.param.nlabel]) # [1024, 10]
        b_fc2 = Hemorrhage_Classification.bias_variable([self.param.nlabel]) # [10]
        
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(y_conv.get_shape)  # -> output: 10
    
        ## Train and Evaluate the Model
        # set up for optimization (optimizer:ADAM)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)  # 1e-4
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())


    def get_filenames(self):    
        try:
            self.subject_2_split = pd.read_csv("list_subjects_test.csv")

        except IOError:
            Hemorrhage_Classification.create_nifti(self)
        
        subject_2_split = pd.read_csv("list_subjects_test.csv")
        df = subject_2_split.rename(index = str, columns = {"Unnamed: 0":"index","0":"path"})
        df1 = pd.DataFrame(df.path.str.split('path:',1).tolist(), columns = ['Patient','Path'])
        df2 = df1.drop('Patient', 1)
        self.df3 = pd.DataFrame(df2.Path.str.split('group:',1).tolist(), columns = ['Path_Clean','Group'])
        ##split the data
        self.list_subj_train, self.list_subj_test, self.list_subj_train_labels, self.list_subj_test_labels = train_test_split(self.df3.iloc[0:], self.df3.iloc[1:], test_size=1-self.param.split, train_size=self.param.split)
        self.list_subj_train, self.list_subj_valid, self.list_subj_train_labels, self.list_subj_valid_labels = train_test_split(self.list_subj_train, self.list_subj_train_labels, test_size=self.param.valid_split, train_size=1-self.param.valid_split)
        ## encode the disease label
        encoder = LabelBinarizer()
        self.list_subj_train_labels_encode = encoder.fit_transform(self.list_subj_train_labels)
        self.list_subj_test_labels_encode = encoder.fit_transform(self.list_subj_test_labels)
        self.list_subj_test_labels_encode = encoder.fit_transform(self.list_subj_valid_labels)
        #strip whitespace from patient path data
        self.list_subj_train = self.list_subj_train.str.strip()
        self.list_subj_valid = self.list_subj_valid.str.strip()
        self.list_subj_test = self.list_subj_test.str.strip()
        
        
    def get_CT_data(self, data_set, data_set_labels):
        
        if len(data_set) == 0: Hemorrhage_Classification.get_filenames() 
        max = len(data_set)
    
        begin = self.batch_index
        end = self.batch_index + self.param.batch_size
    
        if end >= max:
            end = max
            batch_index = 0
    
        x_data = np.array([], np.float32)
        y_data = np.zeros((self.param.batch_size, self.FLAGS.num_class)) # zero-filled list for 'one hot encoding'
        index = 0
    
        for i in range(begin, end):
            
            imagePath = data_set[i]
            CT_orig = nib.load(imagePath)
            CT_data = CT_orig.get_data()
            
            # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
            resized_image = tf.image.resize_images(images=CT_data, size=(self.param.im_size,self.param.im_size), method=1)
    
            image = self.sess.run(resized_image)  # (256,256,40)
            x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
            y_data[index][data_set_labels[i]]  # assign 1 to corresponding column (one hot encoding)
            index += 1
    
        batch_index += self.param.batch_size  # update index for the next batch
        x_data_ = x_data.reshape(self.param.batch_size, self.param.im_size * self.param.im_size * self.param.im_depth)
    
        return x_data_, y_data
    
    def create_nifti(self):
        for group in os.listdir(self.param.path):
            if os.path.isdir(os.path.join(self.param.path, group)):
                for batch in os.listdir(os.path.join(self.param.path, group)):
                    dicom_sorted_path  = os.path.join(self.param.path, group, batch, 'DICOM-SORTED')
                    if os.path.isdir(os.path.join(dicom_sorted_path)):
                        for subj in os.listdir(os.path.join(dicom_sorted_path)):
                            mrn = subj.split('-')[0]
                            if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                self.param.procedure = [x.lower() for x in self.param.procedure]
                                for input_proc in self.param.procedure:
                                    for contrast in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                        if input_proc in contrast.lower():
                                            for proc in os.listdir(os.path.join(dicom_sorted_path, subj, contrast)):
                                                if self.param.imaging_plane in proc:
                                                    if re.findall(r"2.*mm",proc):
                                                        path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
                                                        for fname in os.listdir(path_study):
                                                            if fname.endswith('.nii.gz'):
                                                                nifti_name = fname
                                                                datetime = proc.split('-')[1]
                                                                datetime = datetime.split('_')[0]
                                                                self.list_subjects = self.list_subjects.append(pd.DataFrame({'MRN': [mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))
                                                                break
                                                        
                                                        else:
                                                            datetime = proc.split('-')[1]
                                                            datetime = datetime.split('_')[0]
                                                            #dicom2nifti.dicom_series_to_nifti(path_study, path_study, reorient_nifti=True)
                                                            print("Converting DICOMS for "+subj+" to NIFTI format")
#                                                            path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
                                                            status, output = commands.getstatusoutput('dcm2nii '+path_study)
                                                            if status != 0:
                                                                self.failed_nifti_conv_subjects.append(subj)
                                                            else:
                                                                index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                                index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                                nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
                                                                #src_dcms = glob(path_study+'/*.dcm')
            #                                                    stacks = dcmstack.parse_and_stack(src_dcms)
            #                                                    stack = stacks.values[0]
            #                                                    nii = stack.to_nifti()
            #                                                    nii.to_filename(subj+contrast+proc+'.nii.gz')
                                                                self.list_subjects = self.list_subjects.append(pd.DataFrame({'MRN': [mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))
                                                            
        list_subjects_to_DF = pd.DataFrame(self.list_subjects)
        list_subjects_to_DF.to_csv("list_subjects_test.csv")                                            

def main():
    parser = get_parser()
    param = parser.parse_args()
    classify = Hemorrhage_Classification(param=param)
    classify.get_filenames()
    classify.run_model()

    ##
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()

