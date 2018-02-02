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
                        default='fsl')
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
                        default=True)


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
    #
    for fname in param.list_paths:
        fname_processed = False
        if os.path.isfile(os.path.join(param.ofolder, param.fname_csv)):
            df_tot = pd.read_csv(os.path.join(param.ofolder, param.fname_csv))
            list_fname_processed = [os.path.join(df_tot.iloc[i].path, df_tot.iloc[i].fname) for i in range(len(df_tot))]
            fname_processed = True if fname in list_fname_processed else fname_processed
        #
        if not fname_processed:
            if os.path.isfile(fname):
                path_im = '/'.join(fname.split('/')[:-1])
                fname_im = fname.split('/')[-1]
                # define fslview command
                cmd_fslview = 'fslview'
                if param.fslview_deprecated:
                    cmd_fslview += '_deprecated'
                cmd_fslview += ' '+fname
                if param.boundaries_contrast != '':
                    cmd_fslview += ' -b '+param.boundaries_contrast
                #run fslview
                s, o = commands.getstatusoutput(cmd_fslview)
                label = raw_input("Please label the image ("+fname_im+"): ")
                #
            else:
                path_im = fname
                fname_im = ''
                label = -1
            #
            df_pat = pd.DataFrame({'path': [path_im], 'fname': [fname_im], 'label': [label]})
            if os.path.isfile(os.path.join(param.ofolder, param.fname_csv)):
                df_tot_prev = pd.read_csv(os.path.join(param.ofolder, param.fname_csv))
                df_tot = df_tot_prev.append(df_pat)
                df_tot = df_tot.drop('Unnamed: 0', axis=1)
            else:
                df_tot = df_pat
            df_tot.to_csv(os.path.join(param.ofolder, param.fname_csv))
    #

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