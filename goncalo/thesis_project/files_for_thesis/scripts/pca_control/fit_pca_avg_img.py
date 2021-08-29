'''
Generate PCA reduction vectors
'''
import os
import argparse
import img_control.edit_imgs as ei
import file_control.edit_txt_files as ef
import distance_measure_control.distance_measure as dm
from sklearn.decomposition import PCA

''' Global scope
'''
# Directories
data_set_dir = './data/data_pca' # use the pngs
# Files
results_output = './results/pca_output.txt'

def pca_reduction(img, dimensions):
    ''' PCA dimensionality reduction of img
    '''
    pca = PCA(dimensions) # Choose number of dimensions
    vectorized_img = pca.fit_transform(img)
    
    print(converted_data.shape)
    return vectorized_img

def controller():
    ''' Main control loop
    '''
    results_output = "./pca_vector_results.txt"
    pca_vectorized = []

    # Iterate through dataset and vectorize each element after preprocessing images according to Bai et al.
    for image in os.listdir(data_set_dir):
        if image.endswith(".jpg"):
            X, Y, Z = image.shape
            cx, cy = int(X / 2), int(Y / 2)
            img = ei.crop_image(image, cx, cy, 192)
            img = ei.rescale_intensity(img)

            pca_vectorized.append(pca_reduction(img, 2))

    final_vector = dm.calc_average_vector(pca_vectorized)
    
    ef.write_ouput(final_vector, results_output)
    print('PCA finished...')

def main():
    controller()

if __name__ == '__main__':
    main()