import matplotlib.image as mpimg
import numpy as np
import pydicom
import ntpath
import os
import nibabel as nib
from PIL import Image
from matplotlib import pyplot as plt

# def path_leaf(path):
#     ''' Return file name independently of path shape
#     '''
#     head, tail = ntpath.split(path)
#     return tail or ntpath.basename(head)

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

def show_img(img):
    ''' Plot the image
    '''
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.savefig('myfig.png') # To save the figure

def check_dimension(img):
    ''' Output the dimentions of an image 
    '''
    print(img.shape)

def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2

def crop_image(image, cx, cy, size):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    print(crop.shape)
    return crop

def main():
    for infile in glob.glob("*.jpg"):
        file, ext = os.path.splitext(infile)
        with Image.open(infile) as im:
            X, Y, Z = im.shape
            cx, cy = int(X / 2), int(Y / 2)
            image = crop_image(im, cx, cy, 192)
            # Intensity rescaling
            image = rescale_intensity(image, (1.0, 99.0))
            im.save(file + "JPEG")

if __name__ == '__main__':
    main()