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
    ### TODO: do a parent parser for the data argument that are gonna be used by other (application specific) parsers (for ex in each preprocessing)
    parser = argparse.ArgumentParser(description="Segmentation function based on 2D convolutional neural networks")
    parser.add_argument("-data",
                        help="Data to train and/or test the segmentation on",
                        type=str,
                        dest="path")
    parser.add_argument("-cfolder",
                        help="Name of the contrast folder you're doing the segmentation on. For ex \"T2\" or \"T2_ax\".",
                        type=str,
                        dest="contrast_folder",
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

class Subject():
    def __init__(self, path='', fname_im='', fname_mask='', ori='', type_set=None, im_data=None, mask_data=None, hdr=None, im_data_preproc=None, im_mask_preproc=None):
        self.path = path
        self.fname_im = fname_im
        self.fname_mask = fname_mask
        #
        self.type = type_set
        #
        self.im_data = im_data
        self.mask_data = mask_data
        self.hdr = None
        self.orientation = ori
        #
        self.im_data_preprocessed = im_data_preproc
        self.im_mask_preprocessed = im_mask_preproc
    #
    def __repr__(self):
        to_print = '\nSubject:   '
        to_print += '   path: '+self.path
        to_print += '   fname image: ' + self.fname_im
        to_print += '   fname mask: ' + self.fname_mask
        to_print += '   orientation of im: ' + self.orientation
        to_print += '   used for : ' + str(self.type)
        return to_print

class Segmentation():
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
        #
        self.model_train_hist = None

        K.set_image_data_format('channels_last')  # defined as b/w images throughout

        
    def preprocessing(self):
        #TODO: add progress bar for processing through subjects
        print('-'*30)
        print('Loading data...')
        print('-'*30)
        # get data
        #
        self.list_subjects = get_data(self.param)
        #
        print('-' * 30)
        print('Preprocessing data...')
        print('-' * 30)
        list_im_preprocessed = []
        list_mask_preprocessed = []
        for subj in self.list_subjects:
            # load images
            nib_im_temp = nib.load(os.path.join(subj.path, subj.fname_im))
            subj.im_data = nib_im_temp.get_data()

            if subj.fname_mask != '':
                nib_mask_temp = nib.load(os.path.join(subj.path, subj.fname_mask))
                subj.mask_data = nib_mask_temp.get_data()
            #
            subj.hdr = nib_im_temp.header
            #
            # change orientation to make sure axial slices are in the correct dimension (= axial in the third dimension)
            im_ori, im_reorient = self.reorient(subj.im_data, subj.hdr)
            if subj.fname_mask != '':
                mask_ori, mask_reorient = self.reorient(subj.mask_data, nib_mask_temp.header)
                assert im_ori == mask_ori, "ERROR: image and mask don't have the same orientation"

            subj.orientation = im_ori
            #
            # resample/interpolate to param.im_size with tensorflow -> force interpolation to a square image
            im_resample = self.resample(im_reorient)
            if subj.fname_mask != '':
                mask_resample = self.resample(mask_reorient)
            else:
                mask_resample = None
            #
            #standardize the image intensities
            im_stand = self.standardization(im_resample)
            #
            # add preprocessed images and masks to list and to subject object
            list_im_preprocessed.append(im_stand)
            list_mask_preprocessed.append(mask_resample)
            #
            subj.im_data_preprocessed = im_stand
            subj.mask_data_preprocessed = mask_resample
        #
        #
        # select subjects used for training and testing
        if self.param.split == 0.0:
            self.list_subj_train, list_im_train, list_mask_train = [], [], []
            self.list_subj_test, list_im_test, list_mask_test = self.list_subjects, list_im_preprocessed, list_mask_preprocessed
        elif self.param.split == 1.0:
            self.list_subj_train, list_im_train, list_mask_train = self.list_subjects, list_im_preprocessed, list_mask_preprocessed
            self.list_subj_test, list_im_test, list_mask_test = [], [], []
        else:
            self.list_subj_train, self.list_subj_test, list_im_train, list_im_test, list_mask_train, list_mask_test = train_test_split(self.list_subjects, list_im_preprocessed, list_mask_preprocessed, test_size=1-self.param.split, train_size=self.param.split)
        self.list_subj_test = np.asarray(self.list_subj_test)

        # ignore subjects without mask for training
        i_no_mask_train = [i for i, mask in enumerate(list_mask_train) if mask is None]
        for i in i_no_mask_train:
            print "Subject " + self.list_subj_train[i].path + " doesn't have a mask, ignored for training."
            list_mask_train.pop(i)
            list_im_train.pop(i)
            self.list_subj_train.pop(i)

        if self.param.split != 0.0:
            # select subjects used for training and validation
            self.list_subj_train, self.list_subj_valid, list_im_train, list_im_valid, list_mask_train, list_mask_valid = train_test_split(self.list_subj_train, list_im_train, list_mask_train, test_size=self.param.valid_split, train_size=1-self.param.valid_split)
        else:
            self.list_subj_valid, list_im_valid, list_mask_valid = [], [], []

        if self.param.split != 0.0:
            self.train_imgs_tensor = np.concatenate(list_im_train, axis = 0)
            self.train_masks_tensor = np.concatenate(list_mask_train, axis = 0)
            #
            self.valid_imgs_tensor = np.concatenate(list_im_valid, axis = 0)
            self.valid_masks_tensor = np.concatenate(list_mask_valid, axis = 0)
        #
        self.fname_model = time.strftime("%y%m%d%H%M%S")+'_CNN_model_seg_'+str(self.param.epochs)+'epochs_'+str(self.param.num_layer)+'_layers'
        np.save(self.fname_model+'_test_set.npy', self.list_subj_test)
        #
        f = open(self.fname_model+'_info.txt', 'w')
        #
        f.write("Subjects used for training: \n")
        for subj in self.list_subj_train:
            f.write("\t"+subj.path+"\n")
            subj.type='train'
        #
        f.write("Subjects used for validation: \n")
        for subj in self.list_subj_valid:
            f.write("\t" + subj.path + "\n")
            subj.type = 'valid'
        #
        f.write("Subjects used for testing: \n")
        for subj in self.list_subj_test:
            f.write("\t" + subj.path + "\n")
            subj.type = 'test'
        #
        f.close()

    def reorient(self, image, hdr, put_back=False):
        # change the orientation of an image
        ori = tuple(nib.orientations.io_orientation(hdr.get_best_affine())[:,0]) # 0=R-L, 1=P-A, 2=I-S
        if ori.index(2) != 0:
            if not put_back: # preprocessing
                image_reorient = np.swapaxes(image, 0, ori.index(2))
                image_reorient = image_reorient[..., np.newaxis]
            else: #postprocessing
                image_reorient = np.swapaxes(image, ori.index(2), 0)
        else:
            image_reorient = image
        image_reorient = image_reorient.astype('float32')
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
        # TODO : give possibility to save result elsewhere than in the same folder as original data
        model = self.load_model_seg()

        dict_dc = {'subject': [], 'dice_coeff': []}

        for subj_test in self.list_subj_test:
            # predict segmentation
            subj_seg_data = model.predict(subj_test.im_data_preprocessed, verbose=1)
            # put seg back into original space
            subj_seg_data_post_proc = self.post_processing(subj_seg_data, subj_test)
            # save segmentation imsage
            im_seg = nib.Nifti1Image(subj_seg_data_post_proc, None, subj_test.hdr)
            path_out = add_suffix(subj_test.path+subj_test.fname_im, '_seg_cnn')
            nib.save(im_seg, path_out)
            #
            # compute dice coeff if mask is available
            dict_dc['subject'].append(subj_test.path)
            if subj_test.mask_data is not None:
                dc_subj = dice_coeff_np(subj_test.mask_data.astype(float), subj_seg_data_post_proc.astype(float))
                dict_dc['dice_coeff'].append(dc_subj)
            else:
                dict_dc['dice_coeff'].append(np.nan)
        #
        dataframe_dc = pd.DataFrame(dict_dc, index=range(len(dict_dc['subject'])))
        dataframe_dc.to_csv(self.fname_model+'_dice_coeff_res.csv')


    def load_model_seg(self):
        list_files = glob.glob('*'+str(self.param.epochs)+'*.h5')
        model_file = max(list_files, key=os.path.getctime)
        
        model = load_model(model_file, custom_objects={'dice_coef_loss': self.dice_coef_loss, 'dice_coef': self.dice_coef})
        
        fname_model = model_file.split('.')[-2]
        if os.path.isfile(fname_model+'_test_set.npy'):
            self.list_subj_test = np.load(fname_model+'_test_set.npy')
        elif self.list_subj_test == []:
            self.preprocessing()
        #
        return model

    def post_processing(self, data, subj):
        # reorient data to original orientation
        ori, data_reorient = self.reorient(data, subj.hdr, put_back=True)
        # resample to original dimension
        data_resample = self.resample_mask_to_subj(data_reorient, subj)
        # binarize output seg
        thr = 0.5
        data_resample[data_resample < thr] = 0
        data_resample[data_resample >= thr] = 1

        return data_resample

    def resample_mask_to_subj(self, data, subj, interp=3):
        from skimage.transform import resize
        # get rid of the extra dimension
        if len(data.shape) == 4 and data.shape[3]==1:
            data = data.reshape(data.shape[:-1])
        original_shape = subj.im_data.shape
        new_shape_2d = (original_shape[subj.orientation.index(0)], original_shape[subj.orientation.index(1)])

        data_resample = np.zeros(original_shape)
        for i in range(original_shape[subj.orientation.index(2)]):
            if subj.orientation.index(2) == 0:
                data_resample[i,:,:] = resize(data[i,:,:], new_shape_2d, order=interp)
            if subj.orientation.index(2) == 1:
                data_resample[:,i,:] = resize(data[:,i,:], new_shape_2d, order=interp)
            if subj.orientation.index(2) == 2:
                data_resample[:,:,i] = resize(data[:,:,i], new_shape_2d, order=interp)

        return data_resample



def get_data(param):
    list_subjects = []
    for subj in os.listdir(param.path):
        if os.path.isdir(os.path.join(param.path, subj)):
            path_contrast = os.path.join(param.path, subj, param.contrast_folder)
            fname_im = ''
            fname_mask = ''
            for f in os.listdir(path_contrast):
                if param.mask in f and fname_mask == '':
                    fname_mask = f
                elif param.im in f and fname_im == '':
                    fname_im = f
            if fname_im != '':
                list_subjects.append(Subject(path=path_contrast, fname_im=fname_im, fname_mask=fname_mask))
    return list_subjects



def dice_coeff_np(y_true, y_pred):
    intersection = np.sum(y_true*y_pred)
    dc = 2*intersection /(np.sum(y_true)+np.sum(y_pred))
    return dc

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