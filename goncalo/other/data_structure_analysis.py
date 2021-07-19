import pandas as pd
import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt
import shutil
import nilearn
import tensorflow as tf
from nilearn import plotting
import cv2
import os
import pydicom
from nilearn.image import mean_img
########## Deal with dicoms
from io import BytesIO
from pydicom import read_file
from pydicom.dataset import Dataset
from pydicom.uid import ImplicitVRLittleEndian
import png

data_path = '/home/goncalo/Documents/thesisCode_bitbucket/itc-main-repo/Goncalo/goncalo/data/single_img_feeder/'

def make_jpg():
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            name_file = file.split('.')[0]
            os.system('med2image -i /home/goncalo/Documents/thesisCode_bitbucket/itc-main-repo/Goncalo/goncalo/data/single_img_feeder/' + file + ' -d /home/goncalo/Documents/thesisCode_bitbucket/itc-main-repo/Goncalo/goncalo/data/data_jpg/' + name_file)

def check_data_dimensions():
    files = os.listdir(data_path)
    # Read in the data
    data_all = []
    for data_file in files:
        data = nibabel.load(data_path + data_file).get_data()    
        data = np.rot90(data.squeeze(), 1)
        print(data_file)
        print(data.shape)
        print()

# Doesn't work yet
def check_model_accuracy():
    path_to_model = '/home/goncalo/Documents/UMCG/PROJECTS/ukbb_cardiac/trained_model/FCN_la_2ch'

    latest = tf.train.latest_checkpoint(path_to_model)
    print(latest)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

def convert_dicom_png():

    # Read the dataset
    # ds = read_file(fp)

    # ds.BasicGrayscaleImageSequence[0].file_meta = Dataset()
    # ds.BasicGrayscaleImageSequence[0].file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    # ds.BasicGrayscaleImageSequence[0].is_little_endian = True
    # arr = ds.BasicGrayscaleImageSequence[0].pixel_array

    ds = pydicom.dcmread('/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/DICOMS_RV/002.con')
    shape = ds.pixel_array.shape
    image_2d = ds.pixel_array.astype(float)

    # Convert to float to avoid overflow or underflow losses.
    # image_2d = arr.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    with open('/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/PNG_RV', 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)

def main():
    # check_data_dimensions()
    # check_model_accuracy()
    # make_jpg()
    convert_dicom_png()


if __name__ == '__main__':
    main()
    

'''
Output of image dimension for nifti short-axis from ukbb demo data

sa.nii.gz
(208, 210, 12, 50)

seg_sa.nii.gz
(208, 210, 12, 50)

sa_ES.nii.gz
(208, 210, 12)

seg_sa_ED.nii.gz
(208, 210, 12)

seg_sa_ES.nii.gz
(208, 210, 12)

sa_ED.nii.gz
(208, 210, 12)

Output of image dimension for nifti short-axis from UMCG demo data (A002)
(224, 224, 10)

'''
