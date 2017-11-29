import os
import commands
import argparse
from seg_2D_cnn import get_data, get_parser_data
from utils import *



def get_parser_sc():
    parser_data = get_parser_data()
    #
    parser = argparse.ArgumentParser(description="Preprocessing function for spinal cord data",
                                     parents=[parser_data],
                                     add_help=False)
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
    list_fname_croping_mask = []
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
        fname_mask_sc = os.path.join(subj.path, 'mask_'+subj.fname_im)
        cmd_create_mask = 'sct_create_mask -i '+os.path.join(subj.path, subj.fname_im)+' -p centerline,'+fname_centerline_auto+' -size '+size_mask+' -f box -o '+fname_mask_sc
        status, output = commands.getstatusoutput(cmd_create_mask)
        list_fname_croping_mask.append(fname_mask_sc)
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
    return list_subjects, list_fname_croping_mask


if __name__ == '__main__':
    parser = get_parser_sc()
    param = parser.parse_args()
    preprocessing_spine(param)