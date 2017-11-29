import os
import commands
import argparse
from seg_2D_cnn import get_data
from utils import *



def get_parser():
    parser = argparse.ArgumentParser(description="Preprocessing function for spinal cord data")
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
    parser.add_argument("-c",
                        help="Contrast of the data to segment (used for automatic SC detection).",
                        type=str,
                        dest="contrast",
                        default="t2",
                        choices=['t1', 't2', 't2s', 'dwi'])
    return parser


def preprocessing_spine(param):
    # get list of subjects
    list_subjects = get_data(param)
    #
    for subj in list_subjects:
        # get centerline
        path_out = os.path.join(subj.path, 'centerline')
        cmd_get_centerline = 'sct_get_centerline -i '+os.path.join(subj.path, subj.fname_im)+' -c '+param.contrast+' -ofolder '+path_out
        status, output = commands.getstatusoutput(cmd_get_centerline)
        fname_centerline_auto = os.path.join(path_out, add_suffix(subj.fname_im, '_centerline_optic'))
        #
        # create mask around SC
        size_mask = '40'
        cmd_create_mask = 'sct_create_mask -i '+os.path.join(subj.path, subj.fname_im)+' -p centerline,'+fname_centerline_auto+' -size '+size_mask+' -f box'
        status, output = commands.getstatusoutput(cmd_create_mask)
        fname_mask_sc = os.path.join(subj.path, 'mask_', subj.fname_im)
        #
        # crop image and SC seg (if exists)
        fname_im_out =  add_suffix(subj.fname_im, '_crop')
        cmd_crop_im = 'sct_crop_image -i '+os.path.join(subj.path, subj.fname_im)+' -m '+fname_mask_sc+' -o '+ os.path.join(subj.path, fname_im_out)
        status, output = commands.getstatusoutput(cmd_crop_im)
        subj.fname_im = fname_im_out
        #
        if subj.fname_mask != '':
            fname_mask_out = add_suffix(subj.fname_mask, '_crop')
            cmd_crop_mask = 'sct_crop_image -i ' + os.path.join(subj.path,subj.fname_mask) + ' -m ' + fname_mask_sc + ' -o ' + os.path.join(subj.path, fname_mask_out)
            status, output = commands.getstatusoutput(cmd_crop_mask)
            subj.fname_mask = fname_mask_out
        #
    return list_subjects


if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()
    preprocessing_spine(param)