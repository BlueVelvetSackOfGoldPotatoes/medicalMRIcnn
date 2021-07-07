'''
HOW - Calculate average vectorized image from PCA
TODO: PREPROCESSING MIGHT BE NECESSARY (MOST LIKELY)
TODO: GET DATA SAMPLE OF UKBB DATASET FROM UMCG SERVER
TODO: WRITE SCRIPT
        i) Get data sample from images used in training (bb)
        ii) Calculate average PCA
                a) iterate through images
                b) Vectorize each image using pca
                c) Calculate average PCA vector by averaging all the resulting vectorized images
                        CODE PLAN: Two methods - script inside the data folder - first method: loop through images in main and generate list of PCAs by adding vectorized images as elements, return list element - second method: iterate through list in main of vectorized images and generate average pca, return average pca
'''

import os
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi

def vectorize(img):
    # # Convert image to grayscale
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # # Convert image to binary
    # _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

def calc_average_vector(li_images)

    final_vector = []
    total = len(li_images)

    # val_i should be a list - the vectorized image
    for i, val_i in enumerate(li_images):
        # val_c is each element of the vectorized image
        for c, val_c in enumerate(val_i):
            final_vector[c] =+ val_c

    # Element-wise average
    for i, val_i in enumerate(li_images):
        li_images[i] = val_i / total

def main():
    data_set_dir = 
    pca_vectorized = []

    # iterate through dataset and vectorize each element
    for filename in os.listdir(data_set_dir):
        pca_vectorized.append(vectorize(filename))

    final_vector = calc_average_vector(pca_vectorized)

if __name__ == '__main__':
    main()