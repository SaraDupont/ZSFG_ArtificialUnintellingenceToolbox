#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:32:32 2018

@author: mccoyd2
"""

import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from utils import *
import nibabel as nib
import numpy as np
from sklearn.cross_validation import train_test_split
import skimage
import pandas as pd
from skimage.transform import resize
import commands
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
## local imports
from utils import *


def get_parser_classify():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-data",
                        help="Data to train and/or test the classification on",
                        type=str,
                        dest="path")

    parser.add_argument('--study',
                        help="list of relevant study studies",
                        dest="study",
                        default="",
                        nargs='+')

    parser.add_argument('--series',
                    help="list of image series for image analysis",
                    dest="series",
                    default="",
                    nargs='+')

    parser.add_argument('-output-path',
                    help="Output filename for the resulting model and list of subjects for training, validation and test sets",
                    dest="output_path",
                    default="")
    
    return parser

def get_parser():
    parser_data = get_parser_classify()
    #
    parser = argparse.ArgumentParser(description="Classification function based on 2D convolutional neural networks", parents=[parser_data])
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
                        default=8)
    
    parser.add_argument("-im-size_x",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size_x",
                        default=512)
    
    parser.add_argument("-im-size_y",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size_y",
                        default=256)

    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=6000)

    parser.add_argument("-batch-size",
                    help="Size of batches that make up each epoch.",
                    type=int,
                    dest="batch_size",
                    default=40)
    
    parser.add_argument("-nlabel",
                    help="Number of disease labels that correspond to images.",
                    type=int,
                    dest="nlabel",
                    default=2)

    parser.add_argument("-num_neuron",
                    help="Number of neurons for the fully connected layer.",
                    type=int,
                    dest="num_neuron",
                    default=1500)

    parser.add_argument("-exclude-label",
                    help="Label that should not be used for report classification",
                    type=int,
                    dest="exclude_label",
                    default="NA")

    parser.add_argument("-outcome",
                    help="Name of column in sheet for outcome label",
                    type=str,
                    dest="outcome")

    parser.add_argument("-up-sample",
                    help="Upsample cases to solve sample imbalance issues",
                    type=bool,
                    dest="up_sample",
                    default = True)

    parser.add_argument("-kernalsize",
                    help="size of kernal used in convolutions",
                    type=int,
                    dest="kernalsize",
                    default = 3)

    parser.add_argument("-train-aug",
                help="use keras to randomly augment the training data",
                type=bool,
                dest="train_aug",
                default = True)

    return parser


class Osteomyelitis_Classification():
    
    def __init__(self, param):
        self.param = param 
        self.list_subjects = pd.DataFrame([])
        self.failed_nifti_conv_subjects = []
        self.batch_index_train = 0
        self.batch_index_valid = 0
        self.batch_index_test = 0
        self.list_training_subjects = []
        self.list_training_subjects_labels = []
        self.list_test_subjects = []
        self.list_test_subjects_labels = []
        K.set_image_data_format('channels_last') 
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
        self.x_ = tf.placeholder(tf.float32, shape=[None, self.param.im_size_x*self.param.im_size_x])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.param.nlabel]) 
        self.log_path = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/Medical_Image_2D_CNN_Classification"
        self.test_recall_list = []
        self.test_accuracy_list = []

    def run_model(self):
        
        dic_res_train = {'label_true': [], 'label_pred': [], 'label_pred_prob': []}

        writer_train = tf.summary.FileWriter(self.param.output_path+"/logs/accuracy/training")
        writer_val = tf.summary.FileWriter(self.param.output_path+"/logs/accuracy/validation")

        tf.summary.scalar("accuracy", self.accuracy)
        write_op = tf.summary.merge_all()

        # summary_writer_train_acc = tf.summary.FileWriter(self.param.output_path+"/logs/accuracy/training", self.sess.graph)
        # summary_writer_valid_acc = tf.summary.FileWriter(self.param.output_path+"/logs/accuracy/validation", self.sess.graph)
        # summary_writer_train_rec = tf.summary.FileWriter(self.param.output_path+"/logs/recall/training", self.sess.graph)
        # summary_writer_valid_rec = tf.summary.FileWriter(self.param.output_path+"/logs/recall/validation", self.sess.graph)

        # writer = tf.summary.FileWriter(self.param.output_path+"/logs/accuracy/combined", graph=self.sess.graph)

        # tf.summary.scalar("training_accuracy", self.accuracy)
        # tf.summary.scalar("validation_accuracy", self.accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        # summary_op = tf.summary.merge_all()

        # training_summary_rec = tf.summary.scalar("training_accuracy", self.recall_op)
        # validation_summary_rec = tf.summary.scalar("validation_accuracy", self.recall_op)

        # summary_op = tf.merge_all_summaries()

        # summary_op = tf.summary.merge([tf.summary.merge_all("training_accuracy"),tf.summary.merge_all("validation_accuracy")],collections='merged')

        # test_summary = tf.summary.scalar("test_recall", self.recall_op)
        # x_data_AP_train, x_data_OBL_train, y_data_train NOT used, for training, the generators are used
        x_data_AP_train, x_data_OBL_train, y_data_train = self.get_xray_data(self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels, aug=True)

        x_data_AP_valid, x_data_OBL_valid, y_data_valid = self.get_xray_data(self.list_subj_valid_AP, self.list_subj_valid_OBL, self.list_subj_valid_labels)
        x_data_valid = np.concatenate((x_data_AP_valid, x_data_OBL_valid), axis=2)
        x_data_valid = x_data_valid.reshape(x_data_valid.shape[0], self.param.im_size_x * self.param.im_size_x)

        for i in range(self.param.epochs):
            for ap_gen, obl_gen in self.combine_generators():
                for ap_data in ap_gen:
                    ap_images = ap_data[0]
                    ap_labels = ap_data[1]
                    break
                    # for obl_data in obl_gen:
                for obl_data in obl_gen:
                    obl_images = obl_data[0]
                    obl_labels = obl_data[1]
                    break
                y_1d = np.asarray([y[1] for y in ap_labels])
                unique, counts = np.unique(y_1d, return_counts=True)
                print counts
                break

            x_data_ = np.concatenate((ap_images, obl_images), axis =2)
            print x_data_.shape
            x_data_ = x_data_.reshape(x_data_.shape[0], self.param.im_size_x * self.param.im_size_x)
            x_data_ = np.asarray(x_data_)

            self.train_step.run(feed_dict={self.x_: x_data_, self.y_: ap_labels, self.keep_prob: 0.8})

            if i%2 == 0:
                train_recall, train_accuracy, summary_train, cross_entropy = self.sess.run([self.recall, self.accuracy, write_op, self.cross_entropy], feed_dict={self.x_:x_data_, self.y_: ap_labels, self.keep_prob: 1.0})
                writer_train.add_summary(summary_train, i)
                writer_train.flush()

                # summary_writer_train_rec.add_summary(train_summary_r, i)
                shuffle_ind = np.arange(x_data_valid.shape[0])
                np.random.shuffle(shuffle_ind)
                shuffle_ind = shuffle_ind[:self.param.batch_size] if len(shuffle_ind)>self.param.batch_size else shuffle_ind
                x_data_valid_batch, y_data_valid_batch = x_data_valid[shuffle_ind], y_data_valid[shuffle_ind]

                valid_recall, valid_accuracy, summary_valid = self.sess.run([self.recall, self.accuracy, write_op], feed_dict={self.x_:x_data_valid_batch, self.y_: y_data_valid_batch, self.keep_prob: 1.0})
                writer_val.add_summary(summary_valid, i)
                writer_val.flush()

                # summary_writer_valid_acc.add_summary(valid_summary_a, i)
                # summary_writer_valid_rec.add_summary(valid_summary_r, i)

                print("step %d, training recall %g, validation recall %g, training accuracy %g, validation accuracy %g, cross_entropy %g"%(i, train_recall,valid_recall, train_accuracy, valid_accuracy, cross_entropy))


                if i > 2000:
                    if train_accuracy <= 0.7:
                        train_prediction=tf.cast(tf.argmax(self.y_conv,1), tf.float64)
                        train_prediction_prob=tf.nn.softmax(tf.cast(self.y_conv,tf.float64))

                        train_prediction = np.asarray(train_prediction.eval(feed_dict={self.x_:x_data_, self.keep_prob: 1.0}))
                        train_prediction_prob = np.asarray(train_prediction_prob.eval(feed_dict={self.x_:x_data_, self.keep_prob: 1.0}))

                        [dic_res_train['label_true'].append(label[1]) for label in ap_labels]
                        [dic_res_train['label_pred'].append(label) for label in train_prediction]
                        [dic_res_train['label_pred_prob'].append(label[1]) for label in train_prediction_prob]

        df_train_res = pd.DataFrame(dic_res_train)
        df_train_res.to_csv(self.param.output_path+"/train_results/train_set_results.csv")

        test_df_list = []
        x_data_AP_test, x_data_OBL_test, y_data_test = self.get_xray_data(self.list_subj_test_AP, self.list_subj_test_OBL, self.list_subj_test_labels)
        x_data_test = np.concatenate((x_data_AP_test, x_data_OBL_test), axis=2)
        x_data_test = x_data_test.reshape(x_data_test.shape[0], self.param.im_size_x * self.param.im_size_x)
        start = 0
        end = self.param.batch_size

        dic_res = {'ap_paths':self.list_subj_test_AP, 'obl_paths': self.list_subj_test_OBL, 'true_label':[y[1] for y in y_data_test]}
        list_label_pred = []
        list_label_pred_prob = []

        for i in range(len(self.list_subj_test_AP)/(self.param.batch_size)+1):
            ind_list = range(start, end)
            x_data_test_batch, y_data_test_batch = x_data_test[ind_list], y_data_test[ind_list]
            test_recall = self.recall_op.eval(feed_dict={self.x_: x_data_test_batch, self.y_: y_data_test_batch, self.keep_prob: 1.0})
            test_accuracy = self.accuracy.eval(feed_dict={self.x_: x_data_test_batch, self.y_: y_data_test_batch, self.keep_prob: 1.0})

            prediction=tf.cast(tf.argmax(self.y_conv,1), tf.float64)
            prediction_prob=tf.nn.softmax(tf.cast(self.y_conv, tf.float64))

            test_predictions = np.asarray(prediction.eval(feed_dict={self.x_:x_data_test_batch, self.keep_prob: 1.0}))
            test_predictions_prob = np.asarray(prediction_prob.eval(feed_dict={self.x_:x_data_test_batch, self.keep_prob: 1.0}))

            [list_label_pred.append(label_pred) for label_pred in test_predictions]
            [list_label_pred_prob.append(label_pred_prob[1]) for label_pred_prob in test_predictions_prob]

            self.test_accuracy_list.append(test_accuracy)
            self.test_recall_list.append(test_recall)
            print("test recall %g"%test_recall)
            print("test accuracy %g"%test_accuracy)

            # update indices
            start += self.param.batch_size
            end += self.param.batch_size
            end = x_data_test.shape[0] if end > x_data_test.shape[0] else end
        #
        # test_set_results = pd.concat(test_df_list)
        # test_set_results.columns = ['ap_paths', 'obl_paths', 'prediction class', 'prob negative','prob positive','negative label','positive label']

        dic_res['label_pred'] = list_label_pred
        dic_res['label_pred_prob'] = list_label_pred_prob
        test_set_results = pd.DataFrame(dic_res)

        # train_set_results = pd.concat(train_df_list)
        # train_set_results.columns = ['prediction','negative label','positive label','epoch']
        
        test_set_results.to_csv(self.param.output_path+"/test_results/test_set_results.csv")
        # train_set_results.to_csv(self.param.output_path+"/train_results/train_set_results.csv")
        
        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.sess, self.param.output_path+"/models/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.num_layer)+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_model.ckpt")
        print("Model saved in file: %s" % self.save_path)
        self.test_recall_list = pd.DataFrame(self.test_recall_list)
        self.test_accuracy_list = pd.DataFrame(self.test_accuracy_list)

        self.test_recall_list.to_csv(self.param.output_path+"/test_results/test_recall.csv")
        self.test_accuracy_list.to_csv(self.param.output_path+"/test_results/test_accuracy.csv")
        test_accuracy, test_precision, test_recall, test_f1, test_cm = calc_metrics(test_set_results['true_label'], test_set_results['label_pred'])
        
        print("Test Accuracy " +str(test_accuracy))
        print("Test Precision " +str(test_precision))
        print("Test Recall " +str(test_recall))
        print("Test F1 " +str(test_f1))
        print("Confusion Matrix \n" +str(test_cm))
        
        
    def build_keras_classifer(self):
        # Initialising the CNN
        self.classifier = Sequential()
        
        # Step 1 - Convolution
        self.classifier.add(Conv2D(32, (3, 3), input_shape = (512, 512, 1), activation = 'relu'))
        
        # Step 2 - Pooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a third convolutional layer
        self.classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        
        # Step 3 - Flattening
        self.classifier.add(Flatten())
        
        # Step 4 - Full connection
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dropout(0.5))
        
        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dropout(0.5))
        
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
        # Compiling the CNN
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
        
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

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        input_image = tf.reshape(self.x_, [-1,self.param.im_size_x,self.param.im_size_x,1])

        def tensor_get_shape(tensor):
            s = tensor.get_shape()
            return tuple([s[i].value for i in range(0, len(s))])

        for i in range(self.param.num_layer):
            
            self.list_features.append(self.features)
            self.list_channels.append(self.channels)
            
            print(input_image.get_shape())
            W_conv = weight_variable([self.param.kernalsize, self.param.kernalsize, self.channels, self.features])
            self.list_weight_tensors.append(W_conv)
            b_conv = bias_variable([self.features])
            self.list_bias_tensors.append(b_conv)

            h_conv = tf.nn.relu(conv2d(input_image, W_conv) + b_conv)
            self.list_relu_tensors.append(h_conv)
            print(h_conv.get_shape()) 

            input_image = max_pool_2x2(h_conv)
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
            
        number_neurons = self.param.num_neuron
   
        ## Densely Connected Layer (or fully-connected layer)

        weight_dim1 = last_max_pool_dim[1]*last_max_pool_dim[2]*last_max_pool_dim[3]

        W_fc1 = weight_variable([weight_dim1, number_neurons])

        print(W_fc1.shape)
        b_fc1 = bias_variable([number_neurons])
        
        h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, weight_dim1])

        print(h_pool2_flat.get_shape)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        print(h_fc1.get_shape)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        print(h_fc1_drop.get_shape)
        
        ## Readout Layer
        W_fc2 = weight_variable([self.param.num_neuron, self.param.nlabel])
        b_fc2 = bias_variable([self.param.nlabel])
        
        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(self.y_conv.get_shape)
    
        ## Train and Evaluate the Model
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
       # classes_weights = tf.constant(4.0)
        #weight_per_label = tf.transpose(tf.matmul(self.y_, tf.transpose(classes_weights)) )
        
        
        # ratio = 0.20
        # self.class_weight = tf.constant([ratio, 1.0 - ratio])
        # option 1: pred * weights and then cross entropy on weighted prediction
        # self.weighted_logits = tf.multiply(self.y_conv, self.class_weight) # shape [batch_size, 2]
        # self.xent = tf.nn.softmax_cross_entropy_with_logits(logits = self.weighted_logits, labels = self.y_)
        
        #option 2 : cross entropy of pred and then multiply the cross entropy by weights
#        self.tmp_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.y_conv, labels = self.y_)
#        self.xent = tf.multiply(self.tmp_entropy, self.class_weight)
#        
#        weight_per_label = tf.transpose( tf.matmul(self.y_, tf.transpose(class_weight)) ) #shape [1, batch_size]
#        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits = self.y_conv, labels = self.y_, name="xent_raw")) #shape [1, batch_size]
#         self.cross_entropy = tf.reduce_mean(self.xent)
#         self.cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y_, logits=self.y_conv, pos_weight=classes_weights))
        
        
        
        
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)  # 1e-4
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # recall = tp/(tp+fn)
        self.recall, self.recall_op = tf.metrics.recall(tf.argmax(self.y_, 1), tf.argmax(self.y_conv,1))
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
    def get_xray_data(self, data_set_AP, data_set_OBL, data_set_labels, aug=False):

        max = len(data_set_AP)

        y_data = np.zeros((max, self.param.nlabel))

        x_data_AP = []
        x_data_OBL = []
        
        x_data_failed_AP = []
        x_data_failed_OBL = []
        
        list_dataset_paths_AP = []
        list_dataset_paths_OBL = []
        
        for i in range(max):
            #print("Loading Image %d"%(index))
            imagePath_AP = data_set_AP[i]
            imagePath_OBL = data_set_OBL[i]
            
            try: 
                AP_nifti = nib.load(imagePath_AP)
            except:
                AP_nifti = nib.load(imagePath_AP+'.gz')
            try:
                OBL_nifti = nib.load(imagePath_OBL)
            except:
                OBL_nifti = nib.load(imagePath_OBL+'.gz')
            
            AP_data = AP_nifti.get_data()
            OBL_data = OBL_nifti.get_data()

            ## normalize data
            # plt.imshow(AP_data[:,:,0])
            av_AP, std_AP = np.mean(AP_data), np.std(AP_data)
            AP_data_norm = (AP_data - av_AP)/std_AP
            
            av_OBL, std_OBL = np.mean(OBL_data), np.std(OBL_data)
            OBL_data_norm = (OBL_data - av_OBL)/std_OBL
            
            
            list_dataset_paths_AP.append(imagePath_AP)
            list_dataset_paths_OBL.append(imagePath_OBL)

            # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
            #resized_image = tf.image.resize_images(images=CT_data, size=(self.param.im_size,self.param.im_size,self.param.im_depth), method=1)
            if AP_data.size == 0:
                x_data_failed_AP.append(data_set_AP[i])
                break
            if OBL_data.size == 0: 
                x_data_failed_OBL.append(data_set_OBL[i])

            resized_image_AP = skimage.transform.resize(AP_data_norm, (self.param.im_size_x,self.param.im_size_y,), order=3, mode='reflect')
            resized_image_OBL = skimage.transform.resize(OBL_data_norm, (self.param.im_size_x,self.param.im_size_y), order=3, mode='reflect')

            x_data_AP.append(resized_image_AP)
            x_data_OBL.append(resized_image_OBL)
            
            y_data[i, int(data_set_labels[i])] = 1  # assign 1 to corresponding column (one hot encoding)

        x_data_AP = np.asarray(x_data_AP)
        x_data_OBL = np.asarray(x_data_OBL)

        print 'Augmentation ?', aug
        if aug:
            print 'doing augmentation'
            self.train_datagen_AP = ImageDataGenerator(
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 180,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zca_whitening=False,
                                   width_shift_range= 0.2,
                                   height_shift_range= 0.2)

            self.train_datagen_OBL = ImageDataGenerator(
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 180,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zca_whitening=False,
                                   width_shift_range= 0.2,
                                   height_shift_range= 0.2)

            seed = 12

            # x_data_AP = x_data_AP.reshape(x_data_AP.shape[0], x_data_AP.shape[3], x_data_AP.shape[1], x_data_AP.shape[2])
            # x_data_OBL = x_data_OBL.reshape(x_data_OBL.shape[0], x_data_OBL.shape[3], x_data_OBL.shape[1], x_data_OBL.shape[2])

            self.train_datagen_AP.fit(x_data_AP, seed = seed)
            self.train_datagen_OBL.fit(x_data_OBL, seed = seed)

            self.x_data_AP_generator = self.train_datagen_AP.flow(x_data_AP, y_data, batch_size=self.param.batch_size, shuffle = True, seed=seed)
            self.x_data_OBL_generator = self.train_datagen_OBL.flow(x_data_OBL, y_data, batch_size=self.param.batch_size, shuffle = True, seed=seed)

        return x_data_AP, x_data_OBL, y_data

    def combine_generators(self):
        while True:
            yield(self.x_data_AP_generator,self.x_data_OBL_generator)



    def get_filenames(self):
        quality_control = True

        if not quality_control:
            try:
                self.list_subjs_master = pd.read_csv(self.param.path+"/subject_lists/master_subject_list.csv")

            except IOError:
                self.create_nifti()
                self.list_subjs_master = pd.read_csv(self.param.path+"/subject_lists/master_subject_list.csv")
        
        if quality_control:
            self.list_subjs_master = pd.read_csv(self.param.path+"/subject_lists/master_subject_list.csv")
            testing_QC = pd.read_csv('/media/mccoyd2/hamburger/Osteomyelitis/Data/Quality_Control/images_labels_testing.csv')
            training_QC = pd.read_csv('/media/mccoyd2/hamburger/Osteomyelitis/Data/Quality_Control/images_labels_training.csv')
            validation_QC = pd.read_csv('/media/mccoyd2/hamburger/Osteomyelitis/Data/Quality_Control/images_labels_validation.csv')

            QC_data = testing_QC.append(pd.DataFrame(data = training_QC))
            QC_data = QC_data.append(pd.DataFrame(data = validation_QC))

            QC_data = QC_data.rename(index=str, columns={"accession": "Acn"})

            QC_data_master = pd.merge(self.list_subjs_master,QC_data, on=['Acn'], how = 'inner')
            self.list_subjs_master = QC_data_master[QC_data_master.label != 0]
            self.list_subjs_master = self.list_subjs_master[self.list_subjs_master.label != 2]

            self.list_subjs_master.to_csv(self.param.path+"/subject_lists/Quality_Control_merg.csv")


        def format_date(date):
            expected_len = 14
            if len(str(date)) < expected_len:
                istr = str(date)+'0'*(expected_len-len(str(date)))
                new_i = int(istr)
            else:
                new_i=date
            
            return new_i    
    
        self.list_subjs_master['Datetime'] = self.list_subjs_master['Datetime'].apply(lambda x: format_date(x))
        
        self.list_subjs_master['Datetime_Format'] =  pd.to_datetime(self.list_subjs_master['Datetime'], format='%Y%m%d%H%M%S')
        self.list_subjs_master['Date_Format'] = pd.to_datetime([str(date_time).split(' ')[0] for date_time in self.list_subjs_master['Datetime_Format']])

        x = self.list_subjs_master.groupby(['Acn', 'View Angle Cat']).Datetime_Format.max() 
        y = self.list_subjs_master[self.list_subjs_master['Datetime_Format'].isin(x)]
       
        columns = y.columns.tolist()
        acn_groups = y.groupby(y['Acn'])
    
        datetime_match = pd.DataFrame()
        AP_Images = pd.DataFrame()
        Oblique_Images = pd.DataFrame()
        
        for group in acn_groups: 
            group_df = pd.DataFrame(group[1])   
            if group_df.shape[0] == 2: 
                if group_df['Date_Format'].iloc[0] == group_df['Date_Format'].iloc[1] and group_df['View Angle Cat'].iloc[0] != group_df['View Angle Cat'].iloc[1]:
                    datetime_match = datetime_match.append(group_df)
                    for i in range(group_df.shape[0]):
                        if group_df['View Angle Cat'].iloc[i] == 'AP':
                            AP_Images = AP_Images.append(group_df.iloc[i])
                        if group_df['View Angle Cat'].iloc[i] == 'OBL':
                            Oblique_Images = Oblique_Images.append(group_df.iloc[i])
                    
        self.merged_path_labels_acn_by_line = pd.merge(AP_Images,Oblique_Images, on=['Acn'], how = 'inner')
        
        self.data_from_text_ML = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Results/Predictions.csv')
        self.data_from_radiologist = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Osteomyelitis_Radiologist_Review.csv')
        self.data_from_text_ML_FullApply = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Results/Predictions_Full_Apply.csv')
        
        self.data_labels_radiologist_and_ML = self.data_from_text_ML.append(pd.DataFrame(data = self.data_from_radiologist))
        self.data_labels_radiologist_and_ML_and_Apply = self.data_labels_radiologist_and_ML.append(pd.DataFrame(data = self.data_from_text_ML_FullApply))
#        datetime_match = datetime_match.rename(index=str, columns={"Accession1": "Acn"})
        self.data_labels_radiologist_and_ML_and_Apply = self.data_labels_radiologist_and_ML_and_Apply.rename(index=str, columns={"Accession1": "Acn"})
        
        self.merged_path_labels = pd.merge(self.merged_path_labels_acn_by_line,self.data_labels_radiologist_and_ML_and_Apply, on=['Acn'], how = 'inner')
        
        self.merged_path_labels = self.merged_path_labels[self.merged_path_labels.Osteomyelitis != self.param.exclude_label]
        self.merged_path_labels = self.merged_path_labels[np.isfinite(self.merged_path_labels['Osteomyelitis'])]

        count_labels = self.merged_path_labels.groupby('Osteomyelitis').count()
        print(str(count_labels['Patient_Path_x']))
        
#        merged_label_groups = self.merged_path_labels.groupby(self.merged_path_labels['Acn'])
#        AP_Images = pd.DataFrame()
#        Oblique_Images = pd.DataFrame()
#        
#        columns = self.merged_path_labels.columns.tolist()
#        for group in merged_label_groups: 
#            group = pd.DataFrame(group[1])
#            for i in range(group.shape[0]):
#                if group['View Angle Cat'].iloc[i] == 'AP':
#                    #line = pd.DataFrame(group.iloc[i], columns=columns)
#                    AP_Images = AP_Images.append(group.iloc[i])
#                if group['View Angle Cat'].iloc[i] == 'OBL':
#                    Oblique_Images = Oblique_Images.append(group.iloc[i])
#                    
#        
#        
#        
        self.merged_path_labels = self.merged_path_labels.reset_index(drop=True)
        ##split the data
        self.list_subj_train_AP, self.list_subj_test_AP, self.list_subj_train_OBL, self.list_subj_test_OBL, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test,self.acn_training, self.acn_testing, self.reports_train, self.reports_test = train_test_split(self.merged_path_labels['Patient_Path_x'], self.merged_path_labels['Patient_Path_y'], self.merged_path_labels['Osteomyelitis'], self.merged_path_labels['MRN_y'],self.merged_path_labels['Acn'], self.merged_path_labels['Impression'], test_size=1-self.param.split, train_size=self.param.split)
                
        self.list_subj_train_AP, self.list_subj_valid_AP,self.list_subj_train_OBL,self.list_subj_valid_OBL, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid, self.acn_training, self.acn_valid, self.reports_train, self.reports_valid = train_test_split(self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels, self.mrn_training, self.acn_training,self.reports_train, test_size=self.param.valid_split ,train_size=1-self.param.valid_split)

        self.list_subj_train_labels = self.list_subj_train_labels.values
        self.list_subj_valid_labels = self.list_subj_valid_labels.values
        self.list_subj_test_labels = self.list_subj_test_labels.values
        
        self.list_subj_train_AP = self.list_subj_train_AP.reset_index(drop=True)
        self.list_subj_valid_AP = self.list_subj_valid_AP.reset_index(drop=True)
        self.list_subj_test_AP = self.list_subj_test_AP.reset_index(drop=True)
        
        self.list_subj_train_OBL = self.list_subj_train_OBL.reset_index(drop=True)
        self.list_subj_valid_OBL = self.list_subj_valid_OBL.reset_index(drop=True)
        self.list_subj_test_OBL = self.list_subj_test_OBL.reset_index(drop=True)
        
        self.reports_train = self.reports_train.reset_index(drop=True)
        self.reports_valid = self.reports_valid.reset_index(drop=True)
        self.reports_test = self.reports_test.reset_index(drop=True)
        
        self.mrn_training = self.mrn_training.reset_index(drop = True)
        self.mrn_valid = self.mrn_valid.reset_index(drop = True)
        self.mrn_test = self.mrn_test.reset_index(drop = True)
        
        self.acn_training = self.acn_training.reset_index(drop = True)
        self.acn_valid = self.acn_valid.reset_index(drop = True)
        self.acn_testing = self.acn_testing.reset_index(drop = True)
        
        train = pd.DataFrame({'MRN': self.mrn_training,'Acn': self.acn_training,'Paths_AP':self.list_subj_train_AP,'Paths_OBL':self.list_subj_train_OBL,'Report': self.reports_train,'Labels':self.list_subj_train_labels})
        valid = pd.DataFrame({'MRN': self.mrn_valid,'Acn': self.acn_valid, 'Paths_AP':self.list_subj_valid_AP,'Paths_OBL':self.list_subj_valid_OBL,'Report': self.reports_valid,'Labels':self.list_subj_valid_labels})
        test = pd.DataFrame({'MRN': self.mrn_test,'Acn': self.acn_testing,'Paths_AP':self.list_subj_test_AP, 'Paths_OBL':self.list_subj_test_OBL, 'Report': self.reports_test,'Labels': self.list_subj_test_labels})


        train.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_training_subjects.csv")
        valid.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_validation_subjects.csv")
        test.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_testing_subjects.csv")
        
        print("Training "+ str(np.unique(self.list_subj_train_labels, return_counts=True)))
        print("Validation "+ str(np.unique(self.list_subj_valid_labels, return_counts=True)))
        print("Testing "+ str(np.unique(self.list_subj_test_labels, return_counts=True)))

        if self.param.up_sample == True:
            self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels = self.upsample_minority(self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels, 1, 4)
            self.list_subj_valid_AP, self.list_subj_valid_OBL, self.list_subj_valid_labels = self.upsample_minority(self.list_subj_valid_AP, self.list_subj_valid_OBL, self.list_subj_valid_labels, 1, 4)

            print("Training "+ str(np.unique(self.list_subj_train_labels, return_counts=True)))
            print("Validation "+ str(np.unique(self.list_subj_valid_labels, return_counts=True)))
            print("Testing "+ str(np.unique(self.list_subj_test_labels, return_counts=True)))

        Resampled_df = pd.DataFrame({'Paths_AP':self.list_subj_train_AP,'Paths_OBL':self.list_subj_train_OBL,'Labels':self.list_subj_train_labels})
        Resampled_df.to_csv(self.param.path+"/subject_lists/Resample_QC.csv")

    
    def create_nifti(self):
        self.param.study = [x.lower() for x in self.param.study]
        self.param.series = [x.lower() for x in self.param.series] 
        r = re.compile(".*dcm")
        
        for group in os.listdir(self.param.path):
            if os.path.isdir(os.path.join(self.param.path, group)):
                for batch in os.listdir(os.path.join(self.param.path, group)):
                    dicom_sorted_path  = os.path.join(self.param.path, group, batch, 'DICOM-SORTED')
                    if os.path.isdir(dicom_sorted_path):
                        for subj in os.listdir(dicom_sorted_path):
                            self.mrn = subj.split('-')[0]
                            if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                for study in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                    for param_study in self.param.study: 
                                        if re.findall(param_study, study.lower()):
                                            for series in os.listdir(os.path.join(dicom_sorted_path, subj, study)):
                                                for param_series in self.param.series: 
                                                    if re.findall(param_series, series.lower()):
                                                        path_series = os.path.join(dicom_sorted_path, subj, study, series)
                                                        if len(filter(r.match, os.listdir(path_series))) == 1: 
                                                            nii_in_path = False
                                                            ACN = study.split('-')[0]
                                                            try: 
                                                                datetime = re.findall(r"(\d{14})",series)[0]
                                                            except: 
                                                                datetime = re.findall(r"(\d{8})",study)[0]
                                                            for fname in os.listdir(path_series):    
                                                                if fname.endswith('.nii.gz'):
                                                                    nifti_name = fname
                                                                    nii_in_path = True
            
                                                                    self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN], 'MRN': [self.mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime], 'View Angle': [param_series]}))
                                                                    break
            
                                                            if not nii_in_path:
                                                                ACN = study.split('-')[0]
                                                                print("Converting DICOMS for "+subj+" to NIFTI format")
                                                                status, output = commands.getstatusoutput('dcm2nii '+path_series)
                                                                if status != 0:
                                                                    self.failed_nifti_conv_subjects.append(subj)
                                                                else:
                                                                    index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                                    index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                                    nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
            
                                                                    self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN],'MRN': [self.mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime],  'View Angle': [param_series]}))
        
        list_subjects_to_DF = pd.DataFrame(self.list_subjects)
        list_subjects_to_DF["View Angle Cat"] = np.where(list_subjects_to_DF["View Angle"].str.contains("obl"), "OBL", "AP")
        list_subjects_to_DF.to_csv(self.param.path+"/subject_lists/master_subject_list.csv")
        
    def upsample_minority(self, ap_view_paths, obl_view_paths, labels, target, factor):
        ap_view_paths = np.asarray(ap_view_paths)
        obl_view_paths = np.asarray(obl_view_paths)
        for count, label in enumerate(labels):
            if label == target:
                label_replicate = np.asarray([label]*factor)
                AP_path = ap_view_paths[count]
                OBL_path = obl_view_paths[count]
                AP_path_rep = [AP_path] * factor
                OBL_path_rep = [OBL_path] * factor
                ap_view_paths = np.append(ap_view_paths, AP_path_rep)
                obl_view_paths = np.append(obl_view_paths, OBL_path_rep)
                labels = np.append(labels, label_replicate)
            else:
                pass
        #ap_view_paths.reset_index(drop=True)
        #obl_view_paths.reset_index(drop=True)
        shuffle_ind = np.arange(len(labels))
        np.random.shuffle(shuffle_ind)
        ap_view_paths = ap_view_paths[shuffle_ind]
        obl_view_paths = obl_view_paths[shuffle_ind]
        labels = labels[shuffle_ind]

        return ap_view_paths, obl_view_paths, labels

    def save_network_images_nifti(self, data_group):
        path_list = np.array
        save_path_nifti = '/media/mccoyd2/hamburger/Osteomyelitis/Data/Quality_Control/'
        jpeg_path = '/media/mccoyd2/hamburger/Osteomyelitis/Data/Cases_jpeg/'
        jpeg_path = '/media/mccoyd2/hamburger/Osteomyelitis/Data/Controls_jpeg/'
        
        data = pd.read_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_"+data_group.lower()+"_subjects.csv")
    
        for i in range(data.shape[0]):
            AP_path = data['Paths_AP'][i]
            OBL_path = data['Paths_OBL'][i]
            
            try:
                AP_nifti = nib.load(AP_path)
            except:
                AP_nifti = nib.load(AP_path+'.gz')
            try:
                OBL_nifti = nib.load(OBL_path)
            except:
                OBL_nifti = nib.load(OBL_path+'.gz')
            
            AP_data = AP_nifti.get_data()
            OBL_data = OBL_nifti.get_data()
            
            resized_image_AP = skimage.transform.resize(AP_data, (1000,1000), order=3, mode='reflect')
            resized_image_OBL = skimage.transform.resize(OBL_data, (1000,1000), order=3, mode='reflect')
            
            x_data_ = np.concatenate((resized_image_AP, resized_image_OBL), axis =1)
            
            acn = data['Acn'][i]
            labels = data['Labels'][i]
            
            
            
            new_image = nib.Nifti1Image(x_data_, affine=np.eye(4))
            path_list = np.append(path_list, save_path_nifti+data_group+"/"+str(acn)+"_"+str(labels)+".nii.gz")
            nib.save(new_image, save_path_nifti+data_group+"/"+str(acn)+"_"+str(labels)+".nii.gz")
        np.save(save_path_nifti+data_group+'_List', path_list)

            
            
            

def calc_metrics(true, prediction):
    cm = confusion_matrix(true, prediction)
    TN, FP, FN, TP = confusion_matrix(true, prediction).ravel()
    #print(classification_report(true, prediction))
    Accuracy = float((TP + TN))/float((TP + TN + FP + FN))
    Precision = float(TP) / float((TP + FP))
    Recall = float(TP) / float((TP + FN))
    F1_Score = 2 * float(Precision) * float(Recall) / float((Precision + Recall))
    
    return Accuracy, Precision, Recall, F1_Score, cm

def main():
    parser = get_parser()
    param = parser.parse_args()
    classify = Osteomyelitis_Classification(param=param)
    classify.get_filenames()
    # classify.manual_model_test()
    classify.build_vol_classifier()
   # classify.build_keras_classifer()
    classify.run_model()
    classify.save_network_images_nifti('Testing')
    
    classify.sess.close()

    ##
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()
