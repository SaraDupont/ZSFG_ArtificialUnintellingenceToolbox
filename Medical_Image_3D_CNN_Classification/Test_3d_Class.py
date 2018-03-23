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
import saliency



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
    # TODO : make master list a mandatory arg
    parser.add_argument("-fname-list-subj",
                        help="File name of the csv file containing the master list of the subjects paths, use absolute path",
                        type=str,
                        dest="fname_master_in",
                        default=None)

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
                        default=5000)
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
        self.fname_csv_train = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_training_subjects.csv"
        self.fname_csv_valid = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_validation_subjects.csv"
        self.fname_csv_test = str(self.param.im_size) + "x" + str(self.param.im_depth) + "_" + str(self.param.num_layer) + "_" + str(self.param.batch_size) + "_" + str(self.param.epochs) + "_testing_subjects.csv"
        #
        self.folder_logs = 'logs'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_logs)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_logs))
        #
        self.folder_acc = 'accuracy'
        if not os.path.isdir(os.path.join(self.param.path_out, self.folder_logs, self.folder_acc)):
            os.mkdir(os.path.join(self.param.path_out, self.folder_logs, self.folder_acc))

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
        ## set config to True to look at ressource usage and disk usage
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
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
        epsilon = 1e-3

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

            #h_conv = tf.nn.relu(conv3d(input_image, W_conv) + b_conv)
            h_conv = conv3d(input_image, W_conv)
            batch_mean, batch_var = tf.nn.moments(h_conv,[0])
            BN = tf.nn.batch_normalization(h_conv, batch_mean, batch_var, b_conv, None, epsilon)
            relu_layer = tf.nn.relu(BN)

            self.list_relu_tensors.append(relu_layer)
            print(relu_layer.get_shape())
            input_image = max_pool_4x4(relu_layer)
            self.list_max_pooled_tensors.append(input_image)
            print(input_image.get_shape())

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

        self.list_subjs_master = pd.read_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.param.fname_master_in))

        # name of the columns names in the subjects mastr list:
        col_name_pat_path = 'patient_path'
        col_name_label = 'label'
        col_name_mrn = 'MRN'

        ##split the data
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

        #strip whitespace from patient path data
        self.list_subj_train = list(self.list_subj_train.str.strip()) if len(self.list_subj_train) != 0 else list(self.list_subj_train)
        self.list_subj_valid = list(self.list_subj_valid.str.strip()) if len(self.list_subj_valid) != 0 else list(self.list_subj_valid)
        self.list_subj_test = list(self.list_subj_test.str.strip()) if len(self.list_subj_test) != 0 else list(self.list_subj_test)

        train = pd.DataFrame({col_name_mrn: self.mrn_training,col_name_pat_path:self.list_subj_train,col_name_label:self.list_subj_train_labels})
        valid = pd.DataFrame({col_name_mrn: self.mrn_valid, col_name_pat_path:self.list_subj_valid,col_name_label:self.list_subj_valid_labels})
        test = pd.DataFrame({col_name_mrn: self.mrn_test,col_name_pat_path:self.list_subj_test,col_name_label: self.list_subj_test_labels})

        train.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_train))
        valid.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_valid))
        test.to_csv(os.path.join(self.param.path_out, self.folder_subj_lists, self.fname_csv_test))

        
    def get_CT_data(self, data_set, data_set_labels, batch_index):
        
        max = len(data_set)
        end = batch_index + self.param.batch_size

        begin = batch_index

        if end >= max:
            end = max
            batch_index = 0


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

            if CT_data.size == 0:
                x_data_failed.append(data_set[i])
                break

            resized_image = skimage.transform.resize(CT_data, (self.param.im_size,self.param.im_size,self.param.im_depth), order=3, mode='reflect')

            x_data.append(resized_image)
            y_data[index, data_set_labels[i]] = 1  # assign 1 to corresponding column (one hot encoding)

            index += 1
    
        batch_index += self.param.batch_size  # update index for the next batch
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        x_data_ = x_data.reshape(len(range(begin, end)), self.param.im_size * self.param.im_size * self.param.im_depth)
        return x_data_, y_data, batch_index, list_dataset_paths

    def run_model(self):

        writer_train = tf.summary.FileWriter(os.path.join(self.param.path_out, self.folder_logs, self.folder_acc, "training"))
        writer_val = tf.summary.FileWriter(os.path.join(self.param.path_out, self.folder_logs, self.folder_acc, "validation"))

        tf.summary.scalar("accuracy", self.accuracy)
        write_op = tf.summary.merge_all()

        for i in range(self.param.epochs):
            if self.batch_index_train == 0:
                self.list_subj_train = np.array(self.list_subj_train)
                shuffle_ind = np.arange(len(self.list_subj_train))
                np.random.shuffle(shuffle_ind)
                self.list_subj_train = self.list_subj_train[shuffle_ind]
                self.list_subj_train_labels = self.list_subj_train_labels[shuffle_ind]
                #
            batch_train = self.get_CT_data(self.list_subj_train, self.list_subj_train_labels, self.batch_index_train)
            self.batch_index_train = batch_train[2]
            print("Training batch %d is loaded"%(i))
            if self.batch_index_valid == 0:
                self.list_subj_valid = np.array(self.list_subj_valid)
                shuffle_ind = np.arange(len(self.list_subj_valid))
                np.random.shuffle(shuffle_ind)
                self.list_subj_valid = self.list_subj_valid[shuffle_ind]
                self.list_subj_valid_labels = self.list_subj_valid_labels[shuffle_ind]
                #
            batch_validation = self.get_CT_data(self.list_subj_valid, self.list_subj_valid_labels, self.batch_index_valid)
            self.batch_index_valid = batch_validation[2]
            print("Validation batch %d is loaded"%(i))
            # Logging every 100th iteration in the training process.
            if i%2 == 0:
                #train_accuracy = self.accuracy.eval(feed_dict={self.x_:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                train_accuracy, train_summary, cross_entropy = self.sess.run([self.accuracy, write_op, self.cross_entropy], feed_dict={self.x_:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                writer_train.add_summary(train_summary, i)
                writer_train.flush()

                valid_accuracy, valid_summary = self.sess.run([self.accuracy, write_op], feed_dict={self.x_:batch_validation[0], self.y_: batch_validation[1], self.keep_prob: 1.0})
                writer_val.add_summary(valid_summary, i)
                writer_val.flush()

                print("step %d, training accuracy: %g, validation accuracy: %g, training cross entropy: %g"%(i, train_accuracy,valid_accuracy,cross_entropy))

                if i > 50:
                    if valid_accuracy <= 0.5:
                        print batch_validation[3]

            self.train_step.run(feed_dict={self.x_: batch_train[0], self.y_: batch_train[1], self.keep_prob: 0.9})

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
    # classify.manual_model_test()
    classify.build_vol_classifier()
    if classify.param.split != 0:
        classify.run_model()
    else:
        classify.test_model()
    ##
    classify.sess.close()
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()

