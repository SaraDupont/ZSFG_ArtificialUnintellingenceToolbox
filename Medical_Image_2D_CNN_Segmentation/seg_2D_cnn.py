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
                        help="Number of layers in the expanding and contracting path of the Unet model",
                        type=int,
                        dest="num_layer",
                        default=4)
    parser.add_argument("-imsize",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size",
                        default=128)
    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=5000)
    return parser


class Segmentation():
    def __init__(self, param):
        self.param = param # parser parameters
        self.list_subj = [] # list of paths to each subject
        self.list_fname_im = [] # list of images file names
        self.list_fname_mask = [] # list of masks file names
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
        self.test_imgs_tensor = [] # list of preprocessed images as ndarrays to test model on
        self.test_masks_tensor = [] # list of preprocessed masks as ndarrays to test model on
        #
        self.smooth_dc = 1.
        #
        self.model_train_hist = None

        K.set_image_data_format('channels_last')  # defined as b/w images throughout

        
    def preprocessing(self):
        #TODO: add progress bar for processing through subjects
        print('-'*30)
        print('Loading and preprocessing data...')
        print('-'*30)
        # get data
        for subj in os.listdir(self.param.path):
            path_contrast = os.path.join(self.param.path, subj, self.param.contrast)
            found_im = False
            found_mask = False
            for f in os.listdir(path_contrast):
                if self.param.mask in f and not found_mask:
                    self.list_fname_mask.append(f)
                    found_mask = True
                elif self.param.im in f and not found_im:
                    self.list_fname_im.append(f)
                    found_im = True
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
            im_ori, im_reorient = self.reorient(self.list_im[i], self.list_headers[i])
            mask_ori, mask_reorient = self.reorient(self.list_mask[i], nib_mask_temp.header)
            assert im_ori == mask_ori, "ERROR: image and mask don't have the same orientation"
            self.list_orientation.append(im_ori)
            #
            # resample/interpolate to param.im_size with tensorflow -> force interpolation to a square image
            im_resample = self.resample(im_reorient)
            mask_resample = self.resample(mask_reorient)
            #
            #standardize the image intensities
            im_stand = self.standardization(im_resample)
            #
            # add preprocessed images and masks to list
            list_im_preprocessed.append(im_stand)
            list_mask_preprocessed.append(mask_resample)
        #
        # select subjects used for training and testing
        self.list_subj_train, self.list_subj_test, list_im_train, list_im_test, list_mask_train, list_mask_test = train_test_split(self.list_subj, list_im_preprocessed, list_mask_preprocessed, test_size=1-self.param.split, train_size=self.param.split)
        # select subjects used for training and validation
        self.list_subj_train, self.list_subj_valid, list_im_train, list_im_valid, list_mask_train, list_mask_valid = train_test_split(self.list_subj_train, list_im_train, list_mask_train, test_size=self.param.valid_split, train_size=1-self.param.valid_split)
        
        self.train_imgs_tensor = np.concatenate(list_im_train, axis = 0)
        self.train_masks_tensor = np.concatenate(list_mask_train, axis = 0)

        self.valid_imgs_tensor = np.concatenate(list_im_valid, axis = 0)
        self.valid_masks_tensor = np.concatenate(list_mask_valid, axis = 0)
        
        self.test_imgs_tensor = np.concatenate(list_im_test, axis = 0)
        self.test_masks_tensor = np.concatenate(list_mask_test, axis = 0)
        
        self.fname_model = time.strftime("%y%m%d%H%M%S")+'_CNN_model_seg_'+str(self.param.epochs)+'epochs_'+str(self.param.num_layer)+'_layers'
        np.save(self.fname_model+'_test_imgs.npy', self.test_imgs_tensor)
        np.save(self.fname_model+'_test_masks.npy', self.test_masks_tensor)


    def reorient(self, image, hdr):
        # change the orientation of an image
        ori = tuple(nib.orientations.io_orientation(hdr.get_best_affine())[:,0]) # 0=R-L, 1=P-A, 2=I-S
        if ori.index(2) != 0:
            image_reorient = np.swapaxes(image, 0, ori.index(2))
            image_reorient = image_reorient[..., np.newaxis]
        else:
            image_reorient = image
        image_reorient = image_reorient.astype('float32')
#        image_reorient = image_reorient.astype('int32')
        #
        return ori, image_reorient

    def resample(self, image):
        # run a graph as default to avoid information being added to the same graph over iterations and resulting in an overload
        with tf.Graph().as_default():
            # resample an image to a square of size param.im_size
            image_resample = tf.image.resize_images(image, size=[self.param.im_size, self.param.im_size])
            ## convert from type tensor to numpy array
            image_resample = tf.Session().run(image_resample)
        return image_resample 
    
    def resample_sk(self, image):
        from skimage.transform import resize
        image_resample = np.asarray([resize(im, (self.param.im_size, self.param.im_size)) for im in image])
        return image_resample
    
    def standardization(self, image):
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        image /= std

        return image
  
    def train(self):
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        model = self.get_unet()
        ## save the weights of the model on each epoch in order to run test after aborting
        model_checkpoint = ModelCheckpoint(self.fname_model +'.h5', monitor='loss', save_best_only=False)
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        train_generator = self.image_augmentation()
        self.model_train_hist = model.fit_generator(train_generator, steps_per_epoch=self.train_imgs_tensor.shape[0] / 32, epochs=self.param.epochs,  callbacks=[model_checkpoint], validation_data = (self.valid_imgs_tensor, self.valid_masks_tensor))
    
        df_hist = pd.DataFrame(self.model_train_hist.history)
        df_hist.to_csv(self.fname_model+".csv")
#        model.fit(self.train_imgs_tensor, self.train_masks_tensor, batch_size=32, nb_epoch=self.param.epochs, verbose=1, shuffle=True,
#              validation_split=0.0,
#              callbacks=[model_checkpoint])


    ## based on u-net paper
    def get_unet(self):
        inputs = Input((self.param.im_size, self.param.im_size, 1))
        pool_tmp = inputs
        
        list_down_conv = []
        for i in range(self.param.num_layer):
            pool_tmp, conv_tmp = self.down_layer(layer_input=pool_tmp, curr_layer=i)
            list_down_conv.append(conv_tmp)

        conv = Conv2D(self.param.im_size, (3, 3), activation='relu', padding='same')(pool_tmp) #128
        conv = Dropout(0.2)(conv)
        conv_tmp = Conv2D(self.param.im_size, (3, 3), activation='relu', padding='same')(conv)
    
        for i in range(self.param.num_layer):
            conv_tmp = self.up_layer(prev_conv=conv_tmp, down_conv=list_down_conv[-(i+1)], curr_layer=i)
              
        conv_top = Conv2D(1, (1, 1), activation='sigmoid')(conv_tmp)
    
        model = Model(inputs=[inputs], outputs=[conv_top])
        model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])
    
        return model

    def down_layer(self, layer_input, curr_layer):
        conv = Conv2D(self.param.im_size/(2**(self.param.num_layer-curr_layer)), (3, 3), activation='relu', padding='same')(layer_input) #8
        conv = Dropout(0.2)(conv)
        conv = Conv2D(self.param.im_size/(2**(self.param.num_layer-curr_layer)), (3, 3), activation='relu', padding='same')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool, conv

    def up_layer(self, prev_conv, down_conv, curr_layer):
        up = concatenate([UpSampling2D(size=(2, 2))(prev_conv), down_conv], axis=3)
        conv = Conv2D(self.param.im_size/(2**(self.param.num_layer-(self.param.num_layer-(curr_layer+1)))), (3, 3), activation='relu', padding='same')(up)
        conv = Conv2D(self.param.im_size/(2**(self.param.num_layer-(self.param.num_layer-(curr_layer+1)))), (3, 3), activation='relu', padding='same')(conv)
        return conv

     # define loss functions
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dc = (2. * intersection + self.smooth_dc) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth_dc)
        return dc

    def dice_coef_loss(self, y_true, y_pred):
        return - self.dice_coef(y_true, y_pred)
   
    def image_augmentation(self): 
        #  create two instances with the same arguments
        # create dictionary with the input augmentation values
        data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             #zca_whitening=True,
                             #zca_epsilon=1e-6,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2, 
                             horizontal_flip=True,
                             vertical_flip = True)
        
        ## use this method with both images and masks
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        ## fit the augmentation model to the images and masks with the same seed
        image_datagen.fit(self.train_imgs_tensor, augment=True, seed=seed)
        mask_datagen.fit(self.train_masks_tensor, augment=True, seed=seed)

        
        ## set the parameters for the data to come from (images)
        image_generator = image_datagen.flow(
            self.train_imgs_tensor,
            batch_size=32,
            shuffle=True,
            seed=seed)
        ## set the parameters for the data to come from (masks)
        mask_generator = mask_datagen.flow(
            self.train_masks_tensor,
            batch_size=32,
            shuffle=True,
            seed=seed)
                
        while True:
            yield(image_generator.next(), mask_generator.next())
     
        
    def predict_seg(self):
        model = self.load_model_seg()
        result = model.predict(self.test_imgs_tensor, verbose=1)
        
        print "done"
        plt.imshow(result[10,:,:,0], cmap='gray')
        plt.show()
        print 'showing'
        
        self.dice_spine_test = self.dice_coef(self.test_masks_tensor, result)
        sess = tf.Session()
        self.dice_spine_test = sess.run(self.dice_spine_test)
        print 'Dice Coefficient for the test set numpy tensor is: '+str(self.dice_spine_test)

    def load_model_seg(self):
        list_files = glob.glob('*'+str(self.param.epochs)+'epochs_'+str(self.param.num_layer)+'_layers.h5')
        model_file = max(list_files, key=os.path.getctime)
        
        model = load_model(model_file, custom_objects={'dice_coef_loss': self.dice_coef_loss, 'dice_coef': self.dice_coef})
        
        fname_model = model_file.split('.')[-2]
        self.test_imgs_tensor = np.load(fname_model+'_test_imgs.npy')
        self.test_masks_tensor = np.load(fname_model+'_test_masks.npy')
        
        return model
        

def main():
    parser = get_parser()
    param = parser.parse_args()
    seg = Segmentation(param=param)
    seg.preprocessing()
    if param.split != 0.0:
        seg.train()
    
    if param.split != 1.0:
        seg.predict_seg()
    ##
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()