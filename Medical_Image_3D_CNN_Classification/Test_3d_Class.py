#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:13:35 2017

@author: mccoyd2
"""
from utils import *
import os, glob, re, shutil
import argparse
from utils import *
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import skimage
from keras import backend as K
import pandas as pd
from skimage.transform import resize
import commands

# FLAGS = tf.app.flags.FLAGS
# FLAGS.num_class = 2
# FLAGS.data_dir = '/media/mccoyd2/hamburger/Hemorrhage_Study'


def get_parser_classify():
    # classification parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-path-data",
                        help="Data to train and/or test the classification on",
                        type=str,
                        dest="path")
    parser.add_argument("-path-out",
                        help="Working directory / output path",
                        type=str,
                        dest="path_out",
                        default='./')
    parser.add_argument("-fname-list-subj",
                        help="File name of the csv file containing the master list of the subjects paths, use absolute path",
                        type=str,
                        dest="fname_master_in",
                        default=None)
    # parser.add_argument("-imaging_plane",
    #                     help="Axial, Saggital, Coronal etc.",
    #                     type=str,
    #                     dest="imaging_plane",
    #                     default="")
    # parser.add_argument("-slice_thickness",
    #                     help="Slice thickness of the study",
    #                     type=str,
    #                     dest="slice_thickness",
    #                     default="")
    # parser.add_argument('--nargs',
    #                     help="list of relevant study procedures",
    #                     dest="procedure",
    #                     default="CT_BRAIN_WO_CONTRAST",
    #                     nargs='+')
    # parser.add_argument('-output-filename',
    #                 help="Output filename for the resulting model and list of subjects for training, validation and test sets",
    #                 dest="output_filename",
    #                 default="")
    
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
    parser.add_argument("-im-size",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size",
                        default=512)
    parser.add_argument("-im-depth",
                    help="Depth of the image used in the CNN.",
                    type=int,
                    dest="im_depth",
                    default=80)
    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=1000)
    parser.add_argument("-batch-size",
                    help="Size of batches that make up each epoch.",
                    type=int,
                    dest="batch_size",
                    default=32)
    
    parser.add_argument("-nlabel",
                    help="Number of disease labels that correspond to images.",
                    type=int,
                    dest="nlabel",
                    default=2)

    parser.add_argument("-num_neuron",
                    help="Number of neurons for the fully connected layer.",
                    type=int,
                    dest="num_neuron",
                    default=1024)
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
    
    
class Classification():
    
    def __init__(self, param):
        self.param = param
        #
        if not os.path.isdir(self.param.path_out):
            os.mkdir(self.param.path_out)
        #
        self.folder_subj_lists = 'subject_lists'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_subj_lists)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_subj_lists))
        self.fname_csv_master = 'master_subject_list.csv'
        self.fname_csv_train = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_training_subjects.csv"
        self.fname_csv_valid = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_validation_subjects.csv"
        self.fname_csv_test = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_testing_subjects.csv"
        #
        self.folder_logs = 'logs'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_logs)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_logs))
        #
        self.folder_model = 'models'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_model)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_model))
        self.fname_model = str(self.param.im_size)+"x"+str(self.param.im_depth)+"_"+str(self.param.num_layer)+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_model.ckpt"
        #
        self.folder_results ='results'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_results)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_results))
        self.fname_test_results = 'test_results_accuracy.csv'
        #
        self.list_subjects = pd.DataFrame([])
        self.failed_nifti_conv_subjects = []
        self.batch_index_train = 0
        self.batch_index_valid = 0
        self.batch_index_test = 0
        self.list_training_subjects = []
        self.list_training_subjects_labels = []
        self.list_test_subjects = []
        self.list_test_subjects_labels = []
        K.set_image_data_format('channels_last')  # defined as b/w images throughout
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        self.x_ = tf.placeholder(tf.float32, shape=[None, self.param.im_size*self.param.im_size*self.param.im_depth]) # [None, 28*28]
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.param.nlabel])  # [None, 10]
        self.test_accuracy_list = []
        # Include keep_prob in feed_dict to control dropout rate.


    def build_vol_classifier(self):
        
        self.features = 32
        self.channels = 1 
        self.list_weight_tensors = [] 
        self.list_bias_tensors = []
        self.list_relu_tensors = []
        self.list_max_pooled_tensors = []
        self.list_features = []
        self.list_channels = []

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

        # Pooling: max pooling over 2x2 blocks
        def max_pool_4x4(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
            return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')


        
        input_image = tf.reshape(self.x_, [-1,self.param.im_size,self.param.im_size,self.param.im_depth,1])
        #input_image = tf.reshape(x, [-1,512,512,80,1]) 

        def tensor_get_shape(tensor):
            s = tensor.get_shape()
            return tuple([s[i].value for i in range(0, len(s))])

        for i in range(self.param.num_layer):
            
            self.list_features.append(self.features)
            self.list_channels.append(self.channels)
            
            print(input_image.get_shape())
            W_conv = weight_variable([5, 5, 5, self.channels, self.features])
            self.list_weight_tensors.append(W_conv)
            b_conv = bias_variable([self.features])
            self.list_bias_tensors.append(b_conv)
            h_conv = tf.nn.relu(conv3d(input_image, W_conv) + b_conv)  # conv2d, ReLU(x_image * weight + bias)
            self.list_relu_tensors.append(h_conv)
            print(h_conv.get_shape()) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
            input_image = max_pool_4x4(h_conv)  # apply max-pool
            self.list_max_pooled_tensors.append(input_image)
            print(input_image.get_shape()) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32

            last_max_pool_dim = tensor_get_shape(self.list_max_pooled_tensors[-1])

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
        #W_fc1 = weight_variable([(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features, 1024])  # [7*7*64, 1024]
        weight_dim1 = last_max_pool_dim[1]*last_max_pool_dim[2]*last_max_pool_dim[3]*last_max_pool_dim[4]

        number_neurons = self.param.num_neuron
        W_fc1 = weight_variable([weight_dim1, number_neurons])  # [7*7*64, 1024]

        print(W_fc1.shape)
        b_fc1 = bias_variable([number_neurons]) # [1024]]
        
        #h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, (self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features])  # -> output image: [-1, 7*7*64] = 3136
        h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, weight_dim1])  # -> output image: [-1, 7*7*64] = 3136

        print(h_pool2_flat.get_shape)  # (?, 2621440)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
        print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024
        
        ## Dropout (to reduce overfitting; useful when training very large neural network)
        # We will turn on dropout during training & turn off during testing
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        print(h_fc1_drop.get_shape)  # -> output: 1024
        
        ## Readout Layer
        W_fc2 = weight_variable([1024, self.param.nlabel]) # [1024, 10]
        b_fc2 = bias_variable([self.param.nlabel]) # [10]
        
        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(self.y_conv.get_shape)  # -> output: 10
    
        ## Train and Evaluate the Model
        # set up for optimization (optimizer:ADAM)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)  # 1e-4
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


    def get_filenames(self):    

        if not os.path.isfile(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_master)) and self.param.fname_master_in is None:
            self.create_nifti()
        if self.param.fname_master_in is not None:
            shutil.copy(self.param.fname_master_in, os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_master))

        self.list_subjs_master = pd.read_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_master))

        # name of the columns names in the subjects mastr list:
        col_name_pat_path = 'patient_path'
        col_name_label = 'label'
        col_name_mrn = 'MRN'

        ## additional filtering on datetime (keep only initial study) for Hemorrhage study
        # self.additional_filtering()
        #
        ##split the data
        # self.list_subj_train, self.list_subj_test, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test,self.acn_training, self.acn_testing, self.reports_train, self.reports_test = train_test_split(self.list_subjs_master['Patient_Path'], self.list_subjs_master['Label'], self.list_subjs_master['MRN'],self.list_subjs_master['Acn'], self.list_subjs_master['Impression'], test_size=1-self.param.split, train_size=self.param.split)
        # self.list_subj_train, self.list_subj_valid, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid,self.acn_training, self.acn_valid, self.reports_train, self.reports_valid = train_test_split(self.list_subj_train, self.list_subj_train_labels, self.mrn_training, self.acn_training,self.reports_train, test_size=self.param.valid_split ,train_size=1-self.param.valid_split)

        if self.param.split == 0.0:
            self.list_subj_train, self.list_subj_test, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test = pd.Series([]), self.list_subjs_master[col_name_pat_path], pd.Series([]), self.list_subjs_master[col_name_label], pd.Series([]), self.list_subjs_master[col_name_mrn]
        elif self.param.split == 1.0:
            self.list_subj_train, self.list_subj_test, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test = self.list_subjs_master[col_name_pat_path], pd.Series([]), self.list_subjs_master[col_name_label], pd.Series([]), self.list_subjs_master[col_name_mrn], pd.Series([])
        else:
            self.list_subj_train, self.list_subj_test, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test= train_test_split(self.list_subjs_master[col_name_pat_path], self.list_subjs_master[col_name_label], self.list_subjs_master[col_name_mrn], test_size=1-self.param.split, train_size=self.param.split)

        if self.param.valid_split == 0.0:
            self.list_subj_train, self.list_subj_valid, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid = self.list_subj_train, pd.Series([]), self.list_subj_train_labels, pd.Series([]), self.mrn_training, pd.Series([])
        elif self.param.valid_split == 1.0:
            self.list_subj_train, self.list_subj_valid, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid = pd.Series([]), self.list_subj_train, pd.Series([]), self.list_subj_train_labels, pd.Series([]), self.mrn_training
        else:
            self.list_subj_train, self.list_subj_valid, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid = train_test_split(self.list_subj_train, self.list_subj_train_labels, self.mrn_training, test_size=self.param.valid_split ,train_size=1-self.param.valid_split)

        self.list_subj_train_labels = self.list_subj_train_labels.values
        self.list_subj_valid_labels = self.list_subj_valid_labels.values
        self.list_subj_test_labels = self.list_subj_test_labels.values

        ## encode the disease label
        #encoder = LabelBinarizer()
        #self.list_subj_train_labels_encode = encoder.fit_transform(self.list_subj_train_labels)
        #self.list_subj_test_labels_encode = encoder.fit_transform(self.list_subj_test_labels)
        #self.list_subj_valid_labels_encode = encoder.fit_transform(self.list_subj_valid_labels)

        #self.list_subj_train_labels_encode = self.list_subj_train_labels_encode.reshape((self.list_subj_train_labels_encode.shape[0]))
        #self.list_subj_test_labels_encode = self.list_subj_test_labels_encode.reshape((self.list_subj_test_labels_encode.shape[0]))
        #self.list_subj_valid_labels_encode = self.list_subj_valid_labels_encode.reshape((self.list_subj_valid_labels_encode.shape[0]))


        #strip whitespace from patient path data
        self.list_subj_train = list(self.list_subj_train.str.strip()) if len(self.list_subj_train) != 0 else list(self.list_subj_train)
        self.list_subj_valid = list(self.list_subj_valid.str.strip()) if len(self.list_subj_valid) != 0 else list(self.list_subj_valid)
        self.list_subj_test = list(self.list_subj_test.str.strip()) if len(self.list_subj_test) != 0 else list(self.list_subj_test)

        # train = pd.DataFrame({'MRN': self.mrn_training,'Acn': self.acn_training,'Paths':self.list_subj_train,'Report': self.reports_train,'Label':self.list_subj_train_labels})
        # valid = pd.DataFrame({'MRN': self.mrn_valid,'Acn': self.acn_valid, 'Paths':self.list_subj_valid,'Report': self.reports_valid,'Label':self.list_subj_valid_labels})
        # test = pd.DataFrame({'MRN': self.mrn_test,'Acn': self.acn_testing,'Paths':self.list_subj_test,'Report': self.reports_test,'Label': self.list_subj_test_labels})
        #
        train = pd.DataFrame({col_name_mrn: self.mrn_training,col_name_pat_path:self.list_subj_train,col_name_label:self.list_subj_train_labels})
        valid = pd.DataFrame({col_name_mrn: self.mrn_valid, col_name_pat_path:self.list_subj_valid,col_name_label:self.list_subj_valid_labels})
        test = pd.DataFrame({col_name_mrn: self.mrn_test,col_name_pat_path:self.list_subj_test,col_name_label: self.list_subj_test_labels})

        # self.valid_data_df_for_review = pd.concat(self.list_subj_valid, self.list_subj_valid_labels_encode)
        # self.test_data_df_for_review = pd.concat(self.list_subj_test, self.list_subj_test_labels_encode)

        train.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_train))
        valid.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_valid))
        test.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_test))

    # def additional_filtering(self):
    #     self.list_subjs_master['Datetime_Format'] = pd.to_datetime(self.list_subjs_master['Datetime'],
    #                                                                format='%Y%m%d%H%M%S')
    #     mrn_groups = self.list_subjs_master.groupby(self.list_subjs_master['MRN'])
    #     list_subj_initial_CT = mrn_groups.agg(
    #         lambda x: x.loc[x.Datetime_Format.argmin()])  # to access a specific mrn : list_subj_unique.loc[1948791]
    #
    #     self.data_from_text_ML = pd.read_csv(self.param.path + "/Reports/Hemorrhage_Reports_Batch_1_Predictions.csv")
    #     merged_path_labels = pd.merge(list_subj_initial_CT, self.data_from_text_ML, on=['Acn'], how='inner')
    #
    #     merged_path_labels = merged_path_labels[merged_path_labels.Label != 2]
    #     #
    #     self.list_subjs_master = merged_path_labels

        
    def get_CT_data(self, data_set, data_set_labels, batch_index):
        
        max = len(data_set)
        end = batch_index + self.param.batch_size

        begin = batch_index

        if end >= max:
            end = max
            batch_index = 0
        else:
            batch_index += self.param.batch_size  # update index for the next batch

        #x_data = np.array([], np.float32)
        y_data = np.zeros((len(range(begin, end)), self.param.nlabel)) # zero-filled list for 'one hot encoding'
        x_data = []
        x_data_failed = []
        index = 0
        list_dataset_paths = []

        for i in range(begin, end):
            print("Loading Image %d"%(index))
            imagePath = data_set[i]
            CT_orig = nib.load(imagePath)
            CT_data = CT_orig.get_data()

            list_dataset_paths.append(imagePath)
            # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
            #resized_image = tf.image.resize_images(images=CT_data, size=(self.param.im_size,self.param.im_size,self.param.im_depth), method=1)
            if CT_data.size == 0:
                x_data_failed.append(data_set[i])
                break

            resized_image = skimage.transform.resize(CT_data, (self.param.im_size,self.param.im_size,self.param.im_depth), order=3, mode='reflect')
            #resized_image_stand = self.standardization(resized_image)
            #image = self.sess.run(resized_image)  # (256,256,40)
            #x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
            x_data.append(resized_image)
            y_data[index, data_set_labels[i]] = 1  # assign 1 to corresponding column (one hot encoding)
            #y_data.append(data_set_labels[i])
            index += 1
    
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        x_data_ = x_data.reshape(len(range(begin, end)), self.param.im_size * self.param.im_size * self.param.im_depth)
        return x_data_, y_data, batch_index, list_dataset_paths

    # def standardization(self, image):
    #     mean = np.mean(image)
    #     std = np.std(image)
    #     image -= mean
    #     image /= std
    #
    #     return image

    # def crop_outside_skull(self, CT_data, tol=0):
    #     mask = CT_data>tol
    #     for i in range(CT_data.shape[2]):
    #         CT_data[:,:,i][np.ix_(mask[:,:,i].any(1),mask[:,:i].any(0))]
    #
    # def crop_outside_skull(self, CT_data, tol=0):
    #     # Mask of non-black pixels (assuming image has a single channel).
    #     mask = (CT_data > 0) & (CT_data < 50)
    #     coords = np.empty([CT_data.shape[0], CT_data.shape[1], CT_data.shape[2]])
    #     # Coordinates of non-black pixels.
    #     for i in range(CT_data.shape[2]):
    #         coords = np.argwhere(mask)
    #
    #         # Bounding box of non-black pixels.
    #         x0, y0, z0 = coords.min(axis=0)
    #         x1, y1, z0 = coords.max(axis=0) + 1   # slices are exclusive at the top
    #
    #         # Get the contents of the bounding box.
    #         cropped = image[x0:x1, y0:y1]



#     def create_nifti(self):
#         self.param.procedure = [x.lower() for x in self.param.procedure]
#         for group in os.listdir(self.param.path):
#             if os.path.isdir(os.path.join(self.param.path, group)):
#                 for batch in os.listdir(os.path.join(self.param.path, group)):
#                     dicom_sorted_path  = os.path.join(self.param.path, group, batch, 'DICOM-SORTED')
#                     if os.path.isdir(dicom_sorted_path):
#                         for subj in os.listdir(dicom_sorted_path):
#                             self.mrn = subj.split('-')[0]
#                             if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
#                                 for contrast in os.listdir(os.path.join(dicom_sorted_path, subj)):
#                                     for input_proc in self.param.procedure:
#                                         if input_proc in contrast.lower():
#                                             for proc in os.listdir(os.path.join(dicom_sorted_path, subj, contrast)):
#                                                 if self.param.imaging_plane in proc:
#                                                     if re.findall(r"2.*mm",proc):
#                                                         path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
#                                                         nii_in_path = False
#                                                         ACN = contrast.split('-')[0]
#                                                         for fname in os.listdir(path_study):
#                                                             datetime = re.findall(r"(\d{14})",proc)[0]
#                                                             if fname.endswith('.nii.gz'):
#                                                                 nifti_name = fname
#                                                                 nii_in_path = True
# #                                                                datetime = proc.split('-')[1]
# #                                                                datetime = datetime.split('_')[0]
#                                                                 self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN], 'MRN': [self.mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))
#                                                                 break
#
#                                                         if not nii_in_path:
#                                                             ACN = contrast.split('-')[0]
#                                                             #dicom2nifti.dicom_series_to_nifti(path_study, path_study, reorient_nifti=True)
#                                                             print("Converting DICOMS for "+subj+" to NIFTI format")
# #                                                            path_study = os.path.join(dicom_sorted_path, subj, contrast, proc)
#                                                             status, output = commands.getstatusoutput('dcm2nii '+path_study)
#                                                             if status != 0:
#                                                                 self.failed_nifti_conv_subjects.append(subj)
#                                                             else:
#                                                                 index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
#                                                                 index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
#                                                                 nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
#                                                                 #src_dcms = glob(path_study+'/*.dcm')
#             #                                                    stacks = dcmstack.parse_and_stack(src_dcms)
#             #                                                    stack = stacks.values[0]
#             #                                                    nii = stack.to_nifti()
#             #                                                    nii.to_filename(subj+contrast+proc+'.nii.gz')
#                                                                 self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN],'MRN': [self.mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))
#
#         list_subjects_to_DF = pd.DataFrame(self.list_subjects)
#         list_subjects_to_DF.to_csv(os.path.join(self.param.path, self.folder_subj_lists, self.file_csv_master))
#

    def create_nifti(self):
        ## TODO: change the hardcoded name for the dource CSV file by an argument
        data_info = pd.read_csv(os.path.join(self.param.path, "data_moffitt_ct_findings.csv"))
        len_mrn = 8
        list_subjects = []
        for mrn in data_info['MRN']:
            str_mrn = str(mrn)
            if len(str_mrn) < len_mrn:
                str_mrn = '0' * (len_mrn - len(str_mrn)) + str_mrn
            # for each subject: get the label ## 0="no finding", 1="medical finding", 2="image not usable (artifact or cut FOV)", -1="no image"
            label = int(data_info[data_info.MRN == mrn]['finding_num']) if len(data_info[data_info.MRN == mrn])==1 else int(data_info.loc[data_info[data_info.MRN == mrn].index.values.tolist()[0]]['finding_num'])
            path_ct = os.path.join(self.param.path, str_mrn, 'ct')
            if label in [0, 1] and os.path.isdir(path_ct):
                fname_ct = os.path.join(path_ct, os.listdir(path_ct)[0])
                acn = int(data_info[data_info.MRN == mrn]['AccessionNumber']) if len(data_info[data_info.MRN == mrn])==1 else int(data_info.loc[data_info[data_info.MRN == mrn].index.values.tolist()[0]]['AccessionNumber'])
                list_subjects.append({'Acn': acn, 'MRN': str_mrn, 'Patient_Path': fname_ct, 'Label': label})
        #
        df_subjs = pd.DataFrame(list_subjects)
        #
        df_subjs.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_master))

    def run_model(self):
        summary_writer = tf.summary.FileWriter(os.path.join(self.param.path_out, self.folder_logs, "training"),
                                               self.sess.graph)
        summary_writer2 = tf.summary.FileWriter(os.path.join(self.param.path_out, self.folder_logs, "validation"),
                                                self.sess.graph)
        summary_writer3 = tf.summary.FileWriter(os.path.join(self.param.path_out, self.folder_logs, "testing"),
                                                self.sess.graph)

        training_summary = tf.summary.scalar("training_accuracy", self.accuracy)
        validation_summary = tf.summary.scalar("validation_accuracy", self.accuracy)
        test_summary = tf.summary.scalar("test_accuracy", self.accuracy)

        for i in range(self.param.epochs):

            batch_train = self.get_CT_data(self.list_subj_train, self.list_subj_train_labels,
                                           self.batch_index_train)  # x_data_, y_data, batch_index, list_dataset_paths
            self.batch_index_train = batch_train[2]
            print("Training batch %d is loaded" % (i))
            batch_validation = self.get_CT_data(self.list_subj_valid, self.list_subj_valid_labels,
                                                self.batch_index_valid)
            self.batch_index_valid = batch_validation[2]
            print("Validation batch %d is loaded" % (i))
            # Logging every 100th iteration in the training process.
            if i % 2 == 0:
                # train_accuracy = self.accuracy.eval(feed_dict={self.x_:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                train_accuracy, train_summary = self.sess.run([self.accuracy, training_summary],
                                                              feed_dict={self.x_: batch_train[0],
                                                                         self.y_: batch_train[1], self.keep_prob: 1.0})
                summary_writer.add_summary(train_summary, i)

                valid_accuracy, valid_summary = self.sess.run([self.accuracy, validation_summary],
                                                              feed_dict={self.x_: batch_validation[0],
                                                                         self.y_: batch_validation[1],
                                                                         self.keep_prob: 1.0})
                summary_writer2.add_summary(valid_summary, i)

                print("step %d, training accuracy %g, validation accuracy %g" % (i, train_accuracy, valid_accuracy))

                if i > 50:
                    if valid_accuracy <= 0.5:
                        print batch_validation[3]
            self.train_step.run(feed_dict={self.x_: batch_train[0], self.y_: batch_train[1], self.keep_prob: 0.5})

        # Evaulate our accuracy on the test data
        list_pred_labels = self.test_model()

        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.sess, os.path.join(self.param.path_out, self.folder_model, self.fname_model))
        print("Model saved in file: %s" % self.save_path)

    def test_model(self):
        # TODO: make sure that the test set doesn't include subjects that were used to train the model ? (later when it runs fine)
        if self.param.split == 0:
            # restore model
            saver = tf.train.import_meta_graph(os.path.join(self.param.path_out, self.folder_model, self.fname_model+'.meta'))
            saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(self.param.path_out, self.folder_model)))

        n_test_batches = len(self.list_subj_test) / (self.param.batch_size)
        n_test_batches = n_test_batches + 1 if n_test_batches == 0 else n_test_batches
        list_pred_labels = []
        for i in range(n_test_batches+1):
            testset = self.get_CT_data(self.list_subj_test, self.list_subj_test_labels, self.batch_index_test)
            test_accuracy = self.accuracy.eval(feed_dict={self.x_: testset[0], self.y_: testset[1], self.keep_prob: 1.0})
            self.batch_index_test = testset[2]
            self.test_accuracy_list.append(test_accuracy)
            empty_y = np.zeros((testset[0].shape[0], 2))  ### can be used to replace testset[1] as the test set shouldn't necessarily have labels if we don't want to compute accuracy
            y_pred = self.sess.run(tf.argmax(self.y_conv.eval(feed_dict={self.x_: testset[0], self.y_: empty_y, self.keep_prob: 1.0}), 1))
            [list_pred_labels.append(y) for y in y_pred]
            print("test accuracy %g" % test_accuracy)

        self.test_accuracy_list = pd.DataFrame(self.test_accuracy_list)
        self.test_accuracy_list.to_csv(os.path.join(self.param.path_out, self.folder_results, self.fname_test_results))
        res_pred = pd.DataFrame(
            {'path': self.list_subj_test, 'true_labels': self.list_subj_test_labels, 'pred_labels': list_pred_labels})
        res_pred.to_csv(os.path.join(self.param.path_out, self.folder_results, 'results_test_prediction.csv'))

        return list_pred_labels


def main():
    parser = get_parser()
    param = parser.parse_args()
    classify = Classification(param=param)
    classify.get_filenames()
    #
    classify.build_vol_classifier()
    if classify.param.split != 0:
        classify.run_model()
    else:
        classify.test_model()
    ##
    classify.sess.close()
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()

