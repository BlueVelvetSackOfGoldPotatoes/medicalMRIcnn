# from ..img_control import edit_imgs as ic
from PIL import Image
import os
import numpy as np
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings

# settings.disable_validate_slice_increment()
# settings.disable_validate_slicecount()
# settings.disable_validate_orthogonal()

# from ..file_control import folder_empty as folder_checker

def get_dicom_img(dicom_path):
    ''' Extract image from dicom and return png
    '''
    # Read dicom
    ds = pydicom.dcmread(dicom_path)
    # Extract pixel array
    new_image = ds.pixel_array.astype(float)
    
    # Rescale the image
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0

    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    return final_image

def dicom_to_jpg_dataset(input_path, output_path):
    ''' Iterate through images in input_path and convert dicom to jpg. Save image to output_path.
    '''
    names = get_names_filetype(input_path, 'dcm')
    for name in names:
        image = get_dicom_img(input_path + '/' + name)
        image.save(output_path + '/' + name + '.jpg')

def get_names_filetype(path, filetype):
    ''' Iterates through path and returns list of names of files of type = filetype
    '''
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.' + filetype]:
                names.append(filename)
    return names

def nifti_to_jpg_dataset(input_path, output_path):
    ''' Convert nifti images tp jpg.
    '''

    names = get_names_filetype(input_path, 'gz')
    print(names)
    for name in names:
        os.system('med2image -i "' + input_path + name + '" -d "' + output_path + '/' + name + '" --outputFileType jpg')

def dicom2nifti_gonc(in_dir, out_dir):
    dicom2nifti.convert_directory(in_dir, out_dir)

def main():
    # dicom2nifti_gonc('/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/4/', '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/demo_image/4/sa.nii.gz')

    nifti_to_jpg_dataset('/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/demo_image/5/', '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data/ukbb/bai_ukbb_seg_png')

    # dicom_to_jpg_dataset('/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/4/', '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/4/')

if __name__ == '__main__':
    main()