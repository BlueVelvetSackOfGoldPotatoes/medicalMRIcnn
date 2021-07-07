import os
import re
from os import path

mri_dir = "./RV/"

'''
Crawl folder per patient and copy all dicom files to dicom folder
'''
def main():
    for subdir, dirs, files in os.walk(mri_dir):
        for file in files:
            # Get name of the folder
            folder_name = 'A' + os.path.basename(file)[0:3]

            if os.path.splitext(file)[-1] == '.dcm':
                os.system('cp /home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/masks/RV/{0}/{1} {2}'.format(folder_name, os.path.basename(file), '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/DICOMS_RV'))

if __name__ == '__main__':
    main()