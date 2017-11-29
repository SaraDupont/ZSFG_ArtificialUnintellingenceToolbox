import os
import commands
import argparse
import nibabel as nib
from skimage.morphology import dilation, ball
from scipy.ndimage.filters import gaussian_filter
from spine_data_preproc import get_parser_sc, preprocessing_spine
from utils import *



def get_parser_lesion():
    parser_sc = get_parser_sc()
    #
    parser = argparse.ArgumentParser(description="Preprocessing function for spinal cord Lesions data",
                                     parents=[parser_sc])
    parser.add_argument("-lesions",
                        help="String to look for in the images name.",
                        type=str,
                        dest="lesion",
                        default="_lesion")
    return parser

def preprocessing_lesions(param):
    # get data and preprocess spinal cord
    list_subjects, list_fname_croping_mask = preprocessing_spine(param)
    #
    for i, subj in enumerate(list_subjects):
        # find lesion file
        fname_lesions = ''
        for f in os.listdir(subj.path):
            if param.lesion in f and fname_lesions == '':
                fname_lesions = f
        # crop lesion file
        fname_lesions_out = add_suffix(fname_lesions, '_crop')
        cmd_crop_lesion = 'sct_crop_image -i ' + os.path.join(subj.path,fname_lesions) + ' -m ' + list_fname_croping_mask[i] + ' -o ' + os.path.join(subj.path, fname_lesions_out)
        status, output = commands.getstatusoutput(cmd_crop_lesion)
        #
        # dilate and smooth SC
        im_sc = nib.load(os.path.join(subj.path,subj.fname_mask))
        im_raw = nib.load(os.path.join(subj.path,subj.fname_im))
        #
        # dilate
        size_dil = 2
        data_sc_dil = dilation(im_sc.get_data(), selem=ball(size_dil))
        #
        # smooth
        size_smooth = 2
        sigmas = [size_smooth]*len(data_sc_dil.shape)
        data_smooth = gaussian_filter(data_sc_dil.astype(float), sigmas, order=0, truncate=4.0)
        #
        # mask input image with smoothed SC mask
        data_in_masked = data_smooth*im_raw.get_data()
        #
        # save image
        im_out = nib.Nifti1Image(data_in_masked, None, im_raw.header)
        fname_out = add_suffix(subj.fname_im, "_masked")
        nib.save(im_out, os.path.join(subj.path, fname_out))


if __name__ == '__main__':
    parser = get_parser_lesion()
    param = parser.parse_args()
    preprocessing_lesions(param)