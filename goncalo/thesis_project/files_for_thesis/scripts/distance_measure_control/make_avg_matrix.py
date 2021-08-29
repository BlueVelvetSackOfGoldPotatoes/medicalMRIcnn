import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path_umcg = "/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/4/jpg"

def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2

def standardize(img, umcg=False):
    image = cv2.imread(img)
    if umcg:
        # To match ukbb heart feature positions
        image = ndimage.rotate(image, -75)
    resized = cv2.resize(image, (162, 208))
    # plt.imshow(resized)
    # plt.show()
    rescaled = rescale_intensity(resized, (1, 99))
    # plt.imshow(rescaled)
    # plt.show()
    pixels = rescaled.flatten()
    return pixels

def main():
    list_matrices_ukbb = []
    for file in glob.glob(os.path.join(dir_path,"*.jpg")):
        standardized = standardize(file)
        list_matrices_ukbb.append(standardized)

    avg_ukbb_vector = np.mean(list_matrices_ukbb, axis=0)
    name_tosave = '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/img_matrixes/ukbb_training_avg_vector'
    np.save(str(name_tosave), avg_ukbb_vector)

    # calc Euclidean dst avg for every single ukbb vector used previously:
    dist = 0
    for l in list_matrices_ukbb:
        dist = dist + np.linalg.norm(avg_ukbb_vector-l)

    print('Threshold: avg distance ukbb to their avg is {}'.format(dist/len(list_matrices_ukbb)))

    # Distance of one ukbb image from a different dataset to ukbb avg training
    ukbb_sing_image = '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/1/sa_ED/output-slice002.jpg'
    standardized_sing_ukbb = standardize(ukbb_sing_image)
    dist_sing_ukbb = np.linalg.norm(avg_ukbb_vector-standardized_sing_ukbb)
    print("Distance of different dataset ukbb img:{}".format(dist_sing_ukbb))

    # calc avg distance of umcg images: -----------------------------------------------------
    list_matrices_umcg = []
    for file in glob.glob(os.path.join(dir_path_umcg,"*.jpg")):
        standardized = standardize(file, True)
        list_matrices_umcg.append(standardized)

    avg_umcg_vector = np.mean(list_matrices_umcg, axis=0)
    name_tosave = '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/img_matrixes/umcg_avg_vector'
    np.save(str(name_tosave), avg_umcg_vector)

    # distance between umcg avg vector and ukbb avg vector:
    distance_ukbb_umcg = np.linalg.norm(avg_ukbb_vector-avg_umcg_vector)
    print("Distance of ukbb and umcg avg vectors: {}".format(distance_ukbb_umcg))

    # calc Euclidean dst avg for every single umcg vector used previously:
    dist = 0
    for l in list_matrices_umcg:
        dist = dist + np.linalg.norm(avg_umcg_vector-l)

    print('avg distance of each umcg image to umcg avg vector is:{}'.format(dist/len(list_matrices_umcg)))

    # Calc dist from each singular umcg image to ukbb:
    i = 0
    dist_umcg = 0
    for file in glob.glob(os.path.join(dir_path_umcg,"*.jpg")):
        standardized = standardize(file, True)
        dist_umcg = dist_umcg + np.linalg.norm(avg_ukbb_vector-standardized)
        i = i + 1
    print("Distance of umcg imgs to ukbb:{}".format(dist_umcg/i))

if __name__ == '__main__':
    main()
