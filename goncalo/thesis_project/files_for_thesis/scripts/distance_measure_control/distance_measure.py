''' Methods that calculate the distance measure between two images
'''

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