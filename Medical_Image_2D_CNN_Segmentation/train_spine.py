##
from __future__ import print_function

import os
import sys
from skimage.transform import resize
from skimage.io import imsave
from sklearn.preprocessing import Binarizer
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from spine_data_preproc import load_data, sct_resample
#from pathlib import Path
import nibabel as nib
import tensorflow as tf
import sct_utils as sct 
import matplotlib.pyplot as plt
import os.path
import time
from msct_image import Image
from keras.models import load_model
from keras.callbacks import History 
import pandas as pd 
from keras.utils import plot_model
from scipy import stats

K.set_image_data_format('channels_last')  # defined as b/w images throughout
## you're so \smart!
## for possible image size reduction 
img_rows = 128
img_cols = 128

## for dice
smooth = 1. 

save_weights_path = '/home/mccoyd2/Documents/Spinal_Cord/Codes/weights/'
save_history_df_path = '/home/mccoyd2/Documents/Spinal_Cord/Codes/segmentation_history/'
save_models_path = '/home/mccoyd2/Documents/Spinal_Cord/Codes/models/'
save_figures_path = '/home/mccoyd2/Documents/Spinal_Cord/Codes/figures/'

## used for CNN without image augmentation
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

## other possible evaluation methods include volume error, Hausdorff metric, Jaccard, check later on these...


## loss functions to back propogate the network with (minimize)

def dice_coef_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

## based on u-net paper
def get_unet():
    inputs = Input((img_rows, img_cols, 1))
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

## reduce image size for better processing and reshapet to 4d array for tensorflow

def preprocess(imgs, group, mask = False ):
    if group == 'Train': 
        imgs_p = np.delete(imgs, 0, axis = 2)
    else: 
        imgs_p = imgs
    imgs_p = np.swapaxes(imgs_p,0,2)
    imgs_p = imgs_p[..., np.newaxis]
    imgs_p = imgs_p.astype('float32')
    
    if mask == False: 
        mean = np.mean(imgs_p)  ## not necessary if this can be done in the preprocessing step - look into this
        std = np.std(imgs_p)  
    
        imgs_p -= mean ## perhaps not necessary
        imgs_p /= std
        
        imgs_p = imgs_p.astype('float32')
    else:
        pass
        
    return imgs_p

## augment images and corresponding masks in real time from each training pool directory
## not sure if shear or other augmentation functions work out of the box with keras - need to look at this.
def image_augmentation(imgs, masks): 
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
    image_datagen.fit(imgs, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    
    ## set the parameters for the data to come from (images)
    image_generator = image_datagen.flow(
        imgs,
        batch_size=32,
        shuffle=True,
        seed=seed)
    ## set the parameters for the data to come from (masks)
    mask_generator = mask_datagen.flow(
        masks,
        batch_size=32,
        shuffle=True,
        seed=seed)
    
    # combine generators into one which yields image and masks
    #train_generator = zip(image_generator, mask_generator)
    ## return the train generator for input in the CNN 
    #return image_generator, mask_generator
    
    while True:
        yield(image_generator.next(), mask_generator.next())
        

def train_and_predict(tissue,training_type, percent_valid = 0.2):
    epochs = 5000
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_lesion_mask_train, imgs_spine_mask_train  = load_data(group = 'Train', resample = training_type)

    if tissue == 'spine':
        imgs_mask_train = imgs_spine_mask_train
    elif tissue == 'lesion':
        imgs_mask_train = imgs_lesion_mask_train

    imgs_train_p = preprocess(imgs_train, 'Train', mask = False)
    imgs_mask_train_p = preprocess(imgs_mask_train, 'Train', mask = True) ### not being used if image generator is operating vvvvv
#    imgs_lesion_mask_train_p = preprocess(imgs_lesion_mask_train, 'Train', mask = True) ### not being used if image generator is operating vvvvv
#    imgs_spine_mask_train_p = preprocess(imgs_spine_mask_train, 'Train', mask = True) ### not being used if image generator is operating vvvvv
    
    size_data = imgs_train_p.shape[0]
    ind = np.random.choice(range(size_data), int(size_data*percent_valid), replace=False)
    
    imgs_train_validation_p = imgs_train_p[ind,:,:,:]
    imgs_mask_train_validation_p = imgs_mask_train_p[ind,:,:,:]
#    imgs_train_spine_validation_mask_p = imgs_spine_mask_train_p[ind,:,:,:]
#    imgs_train_lesion_validation_mask_p = imgs_lesion_mask_train_p[ind,:,:,:]
    
    validation_data = (imgs_train_validation_p, imgs_mask_train_validation_p)
#    validation_spine = (imgs_train_validation_p, imgs_train_spine_validation_mask_p)
#    validation_lesion = (imgs_train_validation_p, imgs_train_lesion_validation_mask_p)
    
    imgs_train_p_valid_reduced = np.asarray([v for i, v in enumerate(imgs_train_p) if i not in ind])
    imgs_mask_train_p_valid_reduced = np.asarray([v for i, v in enumerate(imgs_mask_train_p) if i not in ind])
#    imgs_train_spine_mask_p_valid_reduced = np.asarray([v for i, v in enumerate(imgs_spine_mask_train_p) if i not in ind])
#    imgs_train_lesion_mask_p_valid_reduced = np.asarray([v for i, v in enumerate(imgs_lesion_mask_train_p) if i not in ind])


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

#    if tissue == 'spine':
#        train_generator = image_augmentation(imgs_train_p_valid_reduced, imgs_train_spine_mask_p_valid_reduced)
#        hist = model.fit_generator(train_generator,steps_per_epoch=imgs_train_p.shape[0] / 32,epochs=epochs,  callbacks=[model_checkpoint], validation_data = validation_spine)
#    else: 
#        train_generator = image_augmentation(imgs_train_p_valid_reduced, imgs_train_lesion_mask_p_valid_reduced)
#        hist = model.fit_generator(train_generator,steps_per_epoch=imgs_train_p.shape[0] / 32,epochs=epochs,  callbacks=[model_checkpoint], validation_data = validation_lesion)
    
    #init=time.time()
    #print(hist.history.keys())
    
    filepath = save_models_path+model_name
    model.save(filepath)
    
    df_hist = pd.DataFrame(hist.history)
    df_hist.to_csv(save_history_df_path+"history_"+tissue+"_"+str(epochs)+"_epochs_"+training_type+".csv")
    #end = time.time()
    #t = end-init
    #print(t)
    return df_hist

#model_hist = hist_spine_sct
#model_hist = pd.read_csv('/home/mccoyd2/Documents/Spinal_Cord/Codes/segmentation_history/history_spine_1000_epochs_sct.csv')

def plot_model_history(model_hist, tissue = 'lesion'): 
    #model_hist = pd.read_csv(model_hist)
    # summarize history for accuracy
    plt.plot(model_hist['dice_coef'])
    plt.plot(model_hist['val_dice_coef'])
    plt.title('model segmentation metrics for '+tissue)
    plt.ylabel('dice coef')
    plt.xlabel('epoch')
    plt.legend(['dice', 'validation dice'], loc='bottom right')
    Epochs = str(model_hist.shape[0])
    
    plt.savefig(save_figures_path+tissue+'_CNN_segmentation_'+Epochs+'_history.png', dpi=1000)
    plt.show()
    

  ## Test CNN on new test data
def load_segmentation_model(): 
    
    time_stamp_spine = '171101175159'
    time_stamp_lesion = '171102004555'
    model_data_path_spine = '/home/mccoyd2/Documents/Spinal_Cord/Codes/models/'+time_stamp_spine+'_CNN_model_spine_seg_'+training_type+'_'+str(epochs)+'.h5'
    model_data_path_lesion = '/home/mccoyd2/Documents/Spinal_Cord/Codes/models/'+time_stamp_lesion+'_CNN_model_lesion_seg_'+training_type+'_'+str(epochs)+'.h5'

    model_spine = load_model(model_data_path_spine, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model_lesion = load_model(model_data_path_lesion, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    return model_spine, model_lesion 

def test_model(training_type = 'sct', tissue = 'spine'):
    
    test_data_path = '/home/mccoyd2/Documents/Spinal_Cord/Lesion_Data/'
    test_data_path = os.path.join(test_data_path, 'Test')
    
    dice_spine = list()
    dice_spine_propseg = list()
    dice_lesion = list()

    #model.load_weights(model_data_path)
    for patient in os.listdir(test_data_path):
        
        patient_path = os.path.join(test_data_path, patient)
        
        os.chdir(patient_path)
        
        images = os.listdir(patient_path)
        total = len(images) 
         ## because jason didn't crop the images when he processed them        
        if training_type == 'sct':
            
            img_p, img_lesion_mask_p, img_mask_spine_p, im_propseg_p = sct_resample(patient)
            
        if training_type == 'nibabel':
            
            img_p, img_lesion_mask_p, img_mask_spine_p, im_propseg_p = nibabel_resample(patient)
        
        test_img_p = preprocess(img_p, group = 'Test', mask = False)
        gold_std_img_p = preprocess(img_mask_spine_p, group = 'Test',mask = True)
        propseg_img_p = preprocess(im_propseg_p, group = 'Test',mask = True)
        lesion_img_p = preprocess(img_lesion_mask_p, group = 'Test',mask = True)

#        optic_img_p = np.swapaxes(optic_img_p,0,2)
#        optic_img_p = optic_img_p[..., np.newaxis]
        
        
                            
        #mean = np.mean(test_t2_img_p)  # mean for data centering
        #std = np.std(test_t2_img_p)  # std for data normalization
        #test_t2_img_p -= meanimgs_nib
        #test_t2_img_p /= std
        
        model_spine, model_lesion = load_segmentation_model() 
        
        result_spine = model_spine.predict(test_img_p, verbose=1)
        result_lesion = model_lesion.predict(test_img_p, verbose=1)

#        result_nifti = nib.Nifti1Image(result, affine=np.eye(4))
#        gold_std_img_p_nifti = nib.Nifti1Image(gold_std_img_p, affine=np.eye(4))
#        optic_img_p_nifti = nib.Nifti1Image(optic_img_p, affine=np.eye(4))
#        test_t2_img_p_nifti = nib.Nifti1Image(test_t2_img_p, affine=np.eye(4))
#        
#        nib.save(result_nifti, subj_path+'/'+subjs+'_Ax_T2c_CNN_Pred.nii.gz')
#        nib.save(test_t2_img_match_nifti, subj_path+'/'+subjs+'_Ax_T2c_Dim_Match.nii.gz')
#        nib.save(gold_std_img_p_nifti, subj_path+'/'+subjs+'_Ax_T2c_p_GS_4CNN.nii.gz')
#        nib.save(optic_img_p_nifti, subj_path+'/'+subjs+'_Ax_T2c_p_Optic.nii.gz')
#        nib.save(test_t2_img_p_nifti, subj_path+'/'+subjs+'_Ax_T2c_p.nii.gz')
        #nib.save(test_t2_img_p, subj_path+'/'+subj_id+'_Ax_T2c_Resized_Python.nii.gz')

        thr_spine, upper, lower = 0.9, 1, 0
        thr_lesion, upper, lower = 0.5, 1, 0
        

        ## get the dice coefficient for the test image 
        
        
#        result_spine[result_spine<thr] = 0.0
#        result_lesion[result_lesion<thr] = 0.0
        result_spine_niftisave = np.where(result_spine>thr_spine, upper, lower)
        result_lesion_niftisave = np.where(result_lesion>thr_lesion, upper, lower)
        gold_std_img_p_niftisave = np.where(gold_std_img_p>thr_spine, upper, lower)
        lesion_img_p_niftisave = np.where(lesion_img_p>thr_lesion, upper, lower)
        propseg_img_p_niftisave = np.where(propseg_img_p>thr_spine, upper, lower)
#        propseg_img_p[propseg_img_p<thr] = 0.0
        
        

##        gold_std_img_p[gold_std_img_p<thr] = 0.0
#        lesion_img_p[lesion_img_p<thr] = 0.0
#        propseg_img_p[propseg_img_p<thr] = 0.0
        
        dx_spine_CNN = dice_coef(gold_std_img_p, result_spine)
        dx_spine_propseg = dice_coef(gold_std_img_p, propseg_img_p)
        dx_lesion_CNN = dice_coef(lesion_img_p, result_lesion)

        sess = tf.Session()
        dice_spine.append(sess.run(dx_spine_CNN))
        dice_spine_propseg.append(sess.run(dx_spine_propseg))
        dice_lesion.append(sess.run(dx_lesion_CNN))
        
        ## save result image as nifti 
        result_spine_nifti = nib.Nifti1Image(result_spine_niftisave, affine=np.eye(4))
        result_lesion_nifti = nib.Nifti1Image(result_lesion_niftisave, affine=np.eye(4))

        t2_image_nifti = nib.Nifti1Image(test_img_p, affine=np.eye(4))
        propseg_nifti = nib.Nifti1Image(propseg_img_p_niftisave, affine=np.eye(4))
        gold_standard_nifti = nib.Nifti1Image(gold_std_img_p_niftisave, affine=np.eye(4))
        lesion_image_nifti = nib.Nifti1Image(lesion_img_p_niftisave, affine=np.eye(4))


        nib.save(result_spine_nifti, patient+'_Ax_T2c_CNN_Pred.nii.gz')
        nib.save(result_lesion_nifti, patient+'_Ax_T2c_CNN_Pred_lesion.nii.gz')
        nib.save(t2_image_nifti, patient+'_Ax_T2c_cropped_resampled_cropped.nii.gz')
        nib.save(propseg_nifti, patient+'_Ax_T2c_Propseg.nii.gz')
        nib.save(gold_standard_nifti, patient+'_Ax_T2c_Gold_Standard.nii.gz')
        nib.save(lesion_image_nifti, patient+'_Ax_T2c_Gold_Std_Lesion.nii.gz')
        
        return dice, dice_propseg

def plot_model(model):
    plot_model(model, to_file='model.png', show_shapes = True)    
    


def calc_metrics():
    dice_mean_CNN = np.mean(dice_spine)
    dice_std_CNN = np.std(dice_spine)
    dice_mean_propseg = np.mean(dice_spine_propseg)
    dice_std_propseg = np.std(dice_spine_propseg)
    dice_mean_lesion = np.mean(dice_lesion)
    dice_mean_lesion = np.std(dice_lesion)
    
    results_ttest = stats.ttest_ind(dice_spine,dice_spine_propseg)

if __name__ =="__main__":
    type_model = sys.argv[1]                
    hist_spine_sct = train_and_predict(tissue= 'spine',training_type = 'sct')
    hist_spine_nib = train_and_predict(tissue= 'spine',training_type = 'nibabel')
    hist_lesion_sct = train_and_predict(tissue= 'lesion',training_type = 'sct')
    hist_lesion_nib = train_and_predict(tissue= 'lesion',training_type = 'nibabel')
#train_and_predict(type="lesion")                
