''' Methods that calculate the distance measure between two images
'''
import numpy as np

def calc_average_vector(li_images):
    ''' Calculate the average vector over all vectorized images '''
    final_vector = []
    total = len(li_img)

    # val_i should be a list - the vectorized image
    for i, val_i in enumerate(li_img):
        # val_c is each element of the vectorized image
        for c, val_c in enumerate(val_i):
            final_vector[c] =+ val_c

    # Element-wise average
    for i, val_i in enumerate(li_img):
        li_img[i] = val_i / total

    return li_img

def calc_euclidean_distance(vector1, vector2):
    ''' Return euclidean distance between two vectors
    '''
    a = np.array(vector1)
    b = np.array(vector2)

    # Euclidean distance is the l2 norm, since the default value of the ord parameter in numpy.linalg.norm is 2 np.linalg.norm works as the Euclidean distance
    distance = np.linalg.norm(a-b)

    print('distance vector after Euclidean distance:' + distance)
    
    return distance

