import os
import argparse
from utils import *
import nibabel as nib
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
        self.list_headers = [] 
        self.smooth = 1. 
        self.K.set_image_data_format('channels_last')  # defined as b/w images throughout
        self.epochs = 5000
        self.valid_split = 0.2

        
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
            nib_im_temp = nib.load(os.path.join(path_subj, self.list_fname_im[i]))
            self.list_im.append(nib_im_temp.get_data())
            
            nib_mask_temp = nib.load(os.path.join(path_subj, self.list_fname_mask[i]))
            self.list_mask.append(nib_mask_temp.get_data())
            
            self.list_headers.append(nib_im_temp.header)
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
            list_im_preprocessed.append(standardization(im_resample))
            list_mask_preprocessed.append(mask_resample)
        #
        # select subjects used for training and testing
        self.list_subj_train, self.list_subj_test, list_im_train, list_im_test, list_mask_train, list_mask_test = train_test_split(self.list_subj, list_im_preprocessed, list_mask_preprocessed, test_size=1-self.param.split, train_size=self.param.split)
        # select subjects used for training and validation
        self.list_subj_train, self.list_subj_valid, list_im_train, list_im_validation, list_mask_train, list_mask_validation = train_test_split(self.list_subj_train, list_im_train, list_mask_train, test_size=self.valid_split, train_size=1-self.valid_split)

        self.list_im_train = [im.data for im in list_im_train]
        self.list_im_valid = [im.data for im in list_im_validation]
        self.list_im_test = [im.data for im in list_im_test]
        
        self.list_mask_train = [mask.data for mask in list_mask_train]
        self.list_mask_valid = [mask.data for mask in list_mask_validation]
        self.list_mask_test = [mask.data for mask in list_mask_test]
        
        self.train_imgs_tensor = np.concatenate(list_im_train, axis = 0)
        self.train_masks_tensor = np.concatenate(list_mask_train, axis = 0)

        self.valid_imgs_tensor = np.concatenate(train_imgs_tensor, axis = 0)
        self.valid_masks_tensor = np.concatenate(train_masks_tensor, axis = 0)
        
        self.test_imgs_tensor = np.concatenate(list_im_test, axis = 0)
        self.test_masks_tensor = np.concatenate(list_mask_test, axis = 0)
        

    def reorient(self, image):
        # change the orientation of an image
        image_or = np.swapaxes(image,0,2)
        image_or = image_or[..., np.newaxis]
        image_or = image_or.astype('float32')
        # TODO: CHANGE IMAGE ORIENTATION:
        # TODO --> GET CURRENT ORIENTATION AND STORE IT
        # TODO --> CHANGE ORDER OF AXIS, MAYBE HAVE AS A PARAMETER WHAT IS THE ORIENTATION OF SLICES WE WANT TO SEGMENT ?? (I.E. AXIAL, CORONAL OR SAGITAL)
        ori = image_or.shape #... # get image orientation of original
        image_reorient = image_or ## get orientation of reoriented image
        #
        return ori, image_reorient

    def resample(self, image):
        # resample an image to a square of size param.im_size
        image_resample = tf.resample(image, size=self.param.im_size) # TODO: CHANGE SYNTAX HERE
        return image_resample
    
   def standardization(self, image):
       mean = np.mean(image)
       std = np.std(image)  
       image -= mean 
       image /= std
       
       return image
    
    # define loss functions
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return - dice_coef(y_true, y_pred)
    
    ## based on u-net paper
    def get_unet(self):
        inputs = Input((self.param.im_size, self.param.im_size, 1))
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
    
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)
    
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
        model = Model(inputs=[inputs], outputs=[conv10])
    
        model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
        return model
    
    def train_and_predict(self):
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)
        Segmentation.preprocessing()
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        model = get_unet()
        ## save the weights of the model on each epoch in order to run test after aborting
        model_name = time.strftime("%y%m%d%H%M%S")+'_CNN_model_'+tissue+'_seg_'+training_type+'_'+str(epochs)+'.h5'
        model_save_name = save_weights_path+model_name
        model_checkpoint = ModelCheckpoint(model_save_name, monitor='loss', save_best_only=False)
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        train_generator = image_augmentation(imgs_train_p_valid_reduced, imgs_mask_train_p_valid_reduced)
        hist = model.fit_generator(train_generator,steps_per_epoch=imgs_train_p.shape[0] / 32,epochs=epochs,  callbacks=[model_checkpoint], validation_data = validation_data)



def main():
    parser = get_parser()
    param = parser.parse_args()
    print "here"
    seg = Segmentation(param=param)
    seg.preprocessing()



if __name__=="__main__":
    main()