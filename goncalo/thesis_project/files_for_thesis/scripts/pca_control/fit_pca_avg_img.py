''' Calculate average vectorized image from PCA
'''
import os
import argparse
import img_control.edit_imgs as ei
import file_control.edit_txt_files as ef
import file_control.folder_empty as check_folder
import distance_measure_control.distance_measure as dm
from sklearn.decomposition import PCA

''' Global scope
'''
# Directories
data_set_dir = './data/data_pca' # use the pngs
# Files
results_output = './results/pca_output.txt'

def controller():
    ''' Main control loop
    '''
    # Clear files with every run.
    clear_files(results_output)

    pca_vectorized = []

    # iterate through dataset and vectorize each element
    for image in os.listdir(data_set_dir):
        if image.endswith(".jpg"):
            X, Y, Z = image.shape
            cx, cy = int(X / 2), int(Y / 2)
            img = ei.crop_image(image, cx, cy, 192)
            img = ei.rescale_intensity(img)

            pca_vectorized.append(ei.pca_reduction(img, 2))

    final_vector = dm.calc_average_vector(pca_vectorized)

    check_folder.check_folder_empty(results_output)
    ef.write_ouput_to_file(final_vector, results_output)
    print('PCA finished...')

def main():
    controller()

if __name__ == '__main__':
    main()