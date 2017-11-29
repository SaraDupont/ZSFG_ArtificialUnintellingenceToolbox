from __future__ import print_function

import os
import numpy as np
import dicom
import matplotlib.pyplot as plt
import nibabel as nib
from pylab import *
from skimage.io import imsave, imread
from skimage.transform import resize
import sct_utils as sct
from msct_image import Image

data_path = '/home/mccoyd2/Documents/Spinal_Cord/Lesion_Data/'

img_rows = 128
img_cols = 128

def sct_resample(patient): 
    try:
        if not os.path.isfile(patient+'_Ax_T2_centerline_optic.nii.gz'):
            sct.run('sct_get_centerline -i '+patient+'_Ax_T2.nii.gz -c t2')
        if not os.path.isfile('mask_crop.nii.gz'):
            sct.run('sct_create_mask -i '+patient+'_Ax_T2.nii.gz -p centerline,'+patient+'_Ax_T2_centerline_optic.nii.gz -size 35mm -f box -o mask_crop.nii.gz')
    except: 
        if not os.path.isfile(patient+'_Ax_T2_centerline_optic.nii'):
            sct.run('sct_get_centerline -i '+patient+'_Ax_T2.nii -c t2')
        if not os.path.isfile('mask_crop.nii.gz'):
            sct.run('sct_create_mask -i '+patient+'_Ax_T2.nii -p centerline,'+patient+'_Ax_T2_centerline_optic.nii -size 35mm -f box -o mask_crop.nii.gz')
    try:
        if not os.path.isfile(patient+'_Ax_T2c.nii.gz'):
            sct.run('sct_crop_image -i '+patient+'_Ax_T2.nii.gz -m mask_crop.nii.gz -o '+patient+'_Ax_T2c.nii.gz')
    except:
        if not os.path.isfile(patient+'_Ax_T2c.nii.gz'):
            sct.run('sct_crop_image -i '+patient+'_Ax_T2.nii -m mask_crop.nii.gz -o '+patient+'_Ax_T2c.nii.gz')
        
    if not os.path.isfile(patient+'_Ax_T2c_man_seg.nii.gz'):
        sct.run('sct_crop_image -i '+patient+'_Ax_T2_man_seg.nii.gz -m mask_crop.nii.gz -o '+patient+'_Ax_T2c_man_seg.nii.gz')
    if not os.path.isfile(patient+'_Ax_T2c_lesion.nii.gz'):
        sct.run('sct_crop_image -i '+patient+'_Ax_T2_lesion.nii.gz -m mask_crop.nii.gz -o '+patient+'_Ax_T2c_lesion.nii.gz')
    
    ##### 
    im_t2 = Image(patient+'_Ax_T2c.nii.gz')
    nx, ny, nz, nt, px, py, pz, pt = im_t2.dim
    new_pix_dim = '0.2x0.2x'+str(pz)
    
    if not os.path.isfile(patient+'_Ax_T2c_resample.nii.gz'):
        sct.run('sct_resample -i '+patient+'_Ax_T2c.nii.gz -mm '+new_pix_dim+' -o '+patient+'_Ax_T2c_resample.nii.gz')
    if not os.path.isfile(patient+'_Ax_T2c_man_seg_resample.nii.gz'):
        sct.run('sct_resample -i '+patient+'_Ax_T2c_man_seg.nii.gz -mm '+new_pix_dim+' -o '+patient+'_Ax_T2c_man_seg_resample.nii.gz')
    if not os.path.isfile(patient+'_Ax_T2c_lesion_resample.nii.gz'):
        sct.run('sct_resample -i '+patient+'_Ax_T2c_lesion.nii.gz -mm '+new_pix_dim+' -o '+patient+'_Ax_T2c_lesion_resample.nii.gz')
    
    #create mask
    ## TODO recreate centerline ino resampled images to avoid using man_seg
    if not os.path.isfile('mask_'+patient.split('_')[0] + '_Ax_T2c_resample.nii.gz'):
        sct.run('sct_create_mask -i '+patient+'_Ax_T2c_resample.nii.gz -p centerline,'+patient+'_Ax_T2c_man_seg_resample.nii.gz -o mask_'+patient.split('_')[0] + '_Ax_T2c_resample.nii.gz -f box -size 128')
    
    if not os.path.isfile(patient.split('_')[0] + '_Ax_T2c_resample_seg.nii.gz'):
        sct.run('sct_propseg -i '+patient+'_Ax_T2c_resample.nii.gz -c t2')
        
        
    image_lesion_mask = patient.split('_')[0] + '_Ax_T2c_lesion_resample.nii.gz'
    image_t2 = patient.split('_')[0] + '_Ax_T2c_resample.nii'
    image_spine_mask = patient.split('_')[0] + '_Ax_T2c_man_seg_resample.nii.gz'
    image_square_mask = 'mask_'+patient.split('_')[0] + '_Ax_T2c_resample.nii.gz'
    image_propseg = patient.split('_')[0] + '_Ax_T2c_resample_seg.nii.gz'
    
    try: 
        #img = nib.load(os.path.join(patient_path, image_name_real))
        im = Image(image_t2)
    except: 
        #img = nib.load(os.path.join(patient_path, image_name_real+'.gz'))
        im = Image(image_t2+'.gz')
             
    # img = img.get_data()
    
    #img_mask = nib.load(os.path.join(patient_path, image_mask_name))
    im_mask_lesion = Image(image_lesion_mask)
    #img_mask = img_mask.get_data()

    #img_mask_spine = nib.load(os.path.join(patient_path, image_spine_mask))
    im_mask_spine = Image(image_spine_mask)
    #img_mask_spine = img_mask_spine.get_data()
    
    im_mask_sq = Image(image_square_mask)        
    
    im_propseg = Image(image_propseg)
    
    ### CROP AND STACK
    #if patient == '1014':
        #print("hjere") 
#        im.crop_and_stack(im_mask_sq, save=False)
    im.crop_and_stack(im_mask_sq, save=False)
    im_mask_lesion.crop_and_stack(im_mask_sq, save=False)
    im_mask_spine.crop_and_stack(im_mask_sq, save=False)
    im_propseg.crop_and_stack(im_mask_sq, save=False)
    
    img_p = im.data
    img_mask_p = im_mask_lesion.data
    img_mask_spine_p = im_mask_spine.data
    im_propseg_p = im_propseg.data
    
    img_p = img_p[:-1, :-1, :]
    img_lesion_mask_p = img_mask_p[:-1, :-1, :]
    img_mask_spine_p = img_mask_spine_p[:-1, :-1, :]
    im_propseg_p = im_propseg_p[:-1, :-1, :]
    
    return img_p, img_lesion_mask_p, img_mask_spine_p, im_propseg_p
    
def nibabel_resample(patient): 
    
    img_rows = 128
    img_cols = 128
    
    image_mask_name = patient.split('_')[0] + '_Ax_T2c_lesion.nii.gz'
    image_name_real = patient.split('_')[0] + '_Ax_T2c.nii.gz'
    image_spine_mask = patient.split('_')[0] + '_Ax_T2c_man_seg.nii.gz'
    image_square_mask = 'mask_'+patient.split('_')[0] + '_Ax_T2c.nii.gz'
    image_propseg = patient.split('_')[0] + '_Ax_T2c_seg.nii.gz'

    
    img = nib.load(os.path.join(patient_path, image_name_real))
    img_mask_spine = nib.load(os.path.join(patient_path, image_spine_mask))
    img_mask_lesion = nib.load(os.path.join(patient_path, image_mask_name))
    img_propseg = nib.load(os.path.join(patient_path, image_propseg))
    
    img = img.get_data()
    img_mask_spine = img_mask_spine.get_data()
    img_mask_lesion = img_mask_lesion.get_data()
    img_propseg = img_propseg.get_data()
    

    img_p = np.ndarray((img_rows, img_cols, img.shape[2]), dtype=np.uint8)
    for i in range(img.shape[2]):
        img_p[:,:,i] = resize(img[:,:,i], (img_cols, img_rows), preserve_range=True)
#            imgs[:,:,imgs.shape[2]-1] =  img_p[:,:,i]
#        img_p = img_p[..., np.newaxis]


    img_lesion_mask_p = np.ndarray((img_rows, img_cols, img_mask_lesion.shape[2]), dtype=np.uint8)
    for i in range(img_mask_lesion.shape[2]):
        img_lesion_mask_p[:,:,i] = resize(img_mask_lesion[:,:,i], (img_cols, img_rows), preserve_range=True)

#        img_mask_p = imgs_p[..., np.newaxis]

    img_mask_spine_p = np.ndarray((img_rows, img_cols, img_mask_spine.shape[2]), dtype=np.uint8)
    for i in range(img_mask_spine.shape[2]):
        img_mask_spine_p[:,:,i] = resize(img_mask_spine[:,:,i], (img_cols, img_rows), preserve_range=True)
        
    img_propseg_p = np.ndarray((img_rows, img_cols, img_propseg.shape[2]), dtype=np.uint8)
    for i in range(img_propseg.shape[2]):
        img_propseg_p[:,:,i] = resize(img_propseg[:,:,i], (img_cols, img_rows), preserve_range=True)
        
    return img_p, img_lesion_mask_p, img_mask_spine_p, img_propseg_p 
    

def create_data(group = 'Train', resample = 'sct'):
    group_data_path = os.path.join(data_path, group)
    patients = os.listdir(group_data_path)
    total = len(patients) 
    
    imgs = np.empty([img_rows, img_cols,1])
    imgs_lesion_mask = np.empty([img_rows, img_cols,1])
    imgs_spine_mask = np.empty([img_rows, img_cols,1])
    
    i = 0
    print('-'*30)
    print('Creating '+group+' images...')
    print('-'*30)
    
    for patient in patients:
        
        patient_path = os.path.join(group_data_path, patient)
        
        os.chdir(patient_path)
        
        images = os.listdir(patient_path)
        total = len(images) 
         ## because jason didn't crop the images when he processed them
        if resample == 'sct': 
             img_p, img_lesion_mask_p, img_mask_spine_p, im_propseg_p = sct_resample()
            
        
        else:
            
            img_p, img_lesion_mask_p, img_mask_spine_p, im_propseg_p = nibabel_resample()
      
        imgs = np.concatenate((imgs, img_p), axis = 2)       
    
        imgs_lesion_mask = np.concatenate((imgs_lesion_mask, img_lesion_mask_p), axis = 2)       
    
        imgs_spine_mask = np.concatenate((imgs_spine_mask, img_mask_spine_p), axis = 2)       
    
    
    root_path = '/home/mccoyd2/Documents/Spinal_Cord/'
    
    np.save(root_path+'Codes/imgs_'+group+'_'+resample+'.npy', imgs)
    np.save(root_path+'Codes/imgs_mask_'+group+'_'+resample+'.npy', imgs_lesion_mask)
    np.save(root_path+'Codes/imgs_spine_mask'+'_'+group+'_'+resample+'.npy', imgs_spine_mask)
    
        
    print('Saving to .npy files done.')


def load_data(group = 'Train', resample = 'sct'):
    root_path = '/home/mccoyd2/Documents/Spinal_Cord/'
    imgs = np.load(root_path+'Codes/imgs_'+group+'_'+resample+'.npy')
    imgs_lesion_mask = np.load(root_path+'Codes/imgs_mask_'+group+'_'+resample+'.npy')
    imgs_spine_mask = np.load(root_path+'Codes/imgs_spine_mask'+'_'+group+'_'+resample+'.npy')

    return imgs, imgs_lesion_mask, imgs_spine_mask



if __name__ == '__main__':
    create_data(group = 'Train', resample = 'nibabel')
    create_data(group = 'Test', resample = 'nibabel')
    create_data(group = 'Train', resample = 'sct')
    create_data(group = 'Test', resample = 'sct')
    
    imgs_sct, imgs_lesion_mask_sct, imgs_spine_mask_sct = load_data(group = 'Train', resample = 'sct')
    imgs_nib, imgs_lesion_mask_nib, imgs_spine_mask_nib = load_data(group = 'Train', resample = 'nibabel')
