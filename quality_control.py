#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:57:33 2018

@author: mccoyd2
"""

import os, argparse, commands
import nibabel as nib
import numpy as np
from PIL import Image
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description='Quality control function to review nifti images easily. 2D images will be saved as JPEG in the specified output folder. 3D images will be opened individually in fslview and labeled from the terminal after closing fslview.')
    parser.add_argument("-type",
                        help="type of review: fsl = open fslview and write label ni terminal, jpg = save images as jpg (only for 2D)",
                        type=str,
                        dest="type",
                        choices=['fsl', 'jpg'],
                        required=True)
    parser.add_argument("-list-paths",
                        help="numpy file containing the list of paths of the images to review.",
                        type=numpy_file,
                        dest="list_paths",
                        default="")
    parser.add_argument("-ofolder",
                        help="Folder to output the JPEG images (jpg) ot CSV file with assigned labels (fsl)",
                        type=create_folder,
                        dest="ofolder",
                        default="./")
    parser.add_argument('-output-csv',
                        help="Output filename for the CSV file (only for fsl mode).",
                        type=str,
                        dest="fname_csv",
                        default="images_labels.csv")
    parser.add_argument('-b',
                        help="boundaries to set the contrast in fslview (only for fsl mode).\nexample: -b 0,100",
                        type=str,
                        dest="boundaries_contrast",
                        default="")
    parser.add_argument('-dep',
                        help="Is fslview deprecated on your machine ?",
                        type=bool,
                        dest="fslview_deprecated",
                        default = False)


    return parser

def create_folder(path_folder):
    if not os.path.isdir(path_folder):
        try:
            os.mkdir(path_folder)
        except:
            pass
    return path_folder

def numpy_file(fname):
    if os.path.isfile(fname) and '.npy' in fname:
        list_paths = np.load(fname)
    else:
        list_paths = None
        pass
    return list_paths

def save_2d_images(param):
    for fname in param.list_paths:
        if os.path.isfile(fname):
            # get output filename
            fname_out = fname.split('/')[-1].split('.')[0]+'.jpg'
            # load nibabel image
            im = nib.load(fname)
            im_data = im.get_data()
            #  check that image is 2D and reshape it
            assert im_data.shape[2] == 1, 'ERROR: '+fname+' is not a 2D image'
            im_data_2d = im_data.reshape(im_data.shape[:-1])
            # convert to PIL image
            im_pil = Image.fromarray(im_data_2d)
            im_pil = im_pil.convert('RGB')
            # save jpeg
            print 'Saving ', fname_out
            im_pil.save(os.path.join(param.ofolder, fname_out), "JPEG")
        else:
            print 'WARNING: ', fname, 'does not exist.'


def review_images(param):
    list_labels = []
    list_fnames = []
    list_paths_exist = []
    list_paths = param.list_paths
    list_accessions = []

    for path in range(1,len(list_paths)):
        if os.path.isfile(list_paths[path]):
            print str(len(list_paths))
            path_im = '/'.join(list_paths[path].split('/')[:-1])
            fname_im = list_paths[path].split('/')[-1]

            accession = fname_im.split('_')[0]

            # define fslview command
            cmd_fslview = 'fslview'
            if param.fslview_deprecated:
                cmd_fslview += '_deprecated'
            cmd_fslview += ' '+list_paths[path]
            if param.boundaries_contrast != '':
                cmd_fslview += ' -b '+param.boundaries_contrast
            #run fslview
            s, o = commands.getstatusoutput(cmd_fslview)
            label = raw_input("Please label the image ("+fname_im+"): ")
            #
            # store image name and label
            list_paths_exist.append(path_im)
            list_fnames.append(fname_im)
            list_labels.append(label)
            list_accessions.append(accession)
        else:
            list_paths_exist.append(list_paths[path])
            list_fnames.append(list_paths[path])
            list_labels.append('-1')

    #
    # save the result as a CSV file
    df = pd.DataFrame({'path': list_paths_exist, 'fname': list_fnames, 'label': list_labels, 'accession': list_accessions})
    df.to_csv(os.path.join(param.ofolder, param.fname_csv))

def main():
    parser = get_parser()
    param = parser.parse_args()
    #
    if param.type == 'jpg':
        save_2d_images(param)
    elif param.type == 'fsl':
        review_images(param)


if __name__=="__main__":
    main()
