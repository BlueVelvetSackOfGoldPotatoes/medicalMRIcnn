import img_control.edit_imgs as ic
import pillow
import os
import file_control.folder_empty as folder_checker

def dicom_to_jpg_dataset(input_path, output_path):
    ''' Iterate through images in input_path and convert dicom to jpg. Save image to output_path.
    '''
    # If folder isn't empty stop the program. 
    folder_checker.check_folder_empty(output_path)

    names = ic.get_names_filetype(input_path, '.dcm')
    for name in names:
        image = ic.get_dicom_img(input_path + '/' + name)
        image.save(output_path + '/' + name + '.jpg')

def nifti_to_jpg_dataset(input_path, output_path):
    ''' Convert nifti images tp jpg.
    '''
    # If folder isn't empty stop the program. 
    folder_checker.check_folder_empty(output_path)

    names = ic.get_names_filetype(input_path, '.nii.gz')
    for name in names:
        os.system('med2image -i ' + input_path + name + '.nii.gz' + '-d ' + output_path + '/' + name + ' /' + '\n' + '--outputFileType jpg')