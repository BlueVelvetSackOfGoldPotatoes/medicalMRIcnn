Most methods output a tensor. X is a graph: a graph is a data structure that contains a set of tensor objects. Tensors are multi-dimensional arrays of one data type.

# deploy_network.py

## File dependencies
from ukbb_cardiac.common.image_utils imported rescale_intensity - now factored into deploy_network.py.

## Deprecations
### Nibabel
get_data() is deprecated in favor of get_dataf()

## Libraries
import time
import math
import numpy as np
import nibabel as nib
import tensorflow.compat.v1 as tf
from ukbb_cardiac.common.image_utils import rescale_intensity

## Deployment parammeters
seq_name = 'sa'
data_dir = '/vol/bitbucket/wbai/own_work/ukbb_cardiac_demo'
model_path = ''
process_seq = True
save_seg = True
seg4 = False

## Functions
# rescale_intensity()
rescale_intensity(image, thres=(1.0, 99.0))
Rescales the image intensity to the range of [0,1].

### Method flow
Takes an image as an argument and the desired amount of quantiles between 1.0 and 99.0. It then calculates the q-th percentile(s) of the image. Does some computations to floating point image data - the rescaling. then returns the image array that was rescaled.
### Method changes projection
Refactor process_seq and process the two time sequences into two different functions.

Most steps after reading data are the same. These should be a function.

# main()
Starts a tensorflow session and initializes the deployment parameters. Loads the metagraph.
It then either processes a seq (both ES and ED) or each individually.

### Method flow
 If it's a sequence:
    Read it as sa.nii.gz and then read the shape of the image as X Y Z T. 
    Rescale the image by using rescale_intensity.
    Get new shape and retrieve an array of that size filled with 0s.
    Pad image size to be a factor of 16 so that upsample and downsample work with the same image size.
    Process each time frame (t in T) and transpose the image.
    Run session to make a prediction.
    Recover the original image from segmentation by transposing the predicted frame.
    Calculate segmentation time. 
    Calculate ED and ES frames.
    Save segmentation in FLAGS.save_seg in the NIfTI format.
If the images are processed in the different timeframes (process_seq = False), then:
    Images are expected to be in the format sa_ED.nii.gz and sa_ES.nii.gz.
    Difference is only on processing input data. The same process in between.

If it s a sequence shows a mean table time averaged per sequence as well as producing the segmented images. If it's two frames, ED and ES, produces an average segmentation time per frame as well as the segmented images.

###############################################

# network.py
## File dependencies
None.

## Deprecations

## Libraries
import tensorflow.compat.v1 as tf
import numpy as np

## Deployment parammeters
None.

## Function flow and dependency word diagram
### Summary
Methods that build layers are dynamically called from train_network.py

## Functions
# conv2d_bn_relu(x, filters, training, kernel_size=3, strides=1)
### Method flow
This method is not used in the file - it's used in train_network.py.

Layer with convolution, batch normalization and rectified linear unit (ReLU function is an activation function defined as the positive part of its argument). Returns tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(...), training=training)).

# conv2d_transpose_bn_relu(x, filters, training, kernel_size=3, strides=1)
### Method flow
This method is not used in the file - it's used in train_network.py.

Layer with 2D convolution (input and output with 2D data), batch normalization and rectified linear unit (ReLU function is an activation function defined as the positive part of its argument). Returns tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(...), training=training)).

# conv3d_bn_relu(x, filters, training, kernel_size=3, strides=1)
### Method flow
This method is not used in the file - it's used in train_network.py.

Layer with 3D convolution (input and output with 3D data), batch normalization and rectified linear unit (ReLU function is an activation function defined as the positive part of its argument). Returns tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d(...), training=training)).

# conv3d_transpose_bn_relu(x, filters, training, kernel_size=3, strides=1)
### Method flow
This method is not used in the file - it's used in train_network.py.

Layer with 3D convolution (input and output with 3D data), batch normalization and rectified linear unit (ReLU function is an activation function defined as the positive part of its argument). Returns tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d_transpose(...), training=training)).

# residual_unit(x, filters, training, strides=1)
Called in build_ResNet and used if use_bottleneck is set to false.
Batch normalization followed by relu and convolutiond 2d. It also handles the skipping or doing convolution 2d a second time.

# bottleneck_unit(x, filters, training, strides=1)
3 times batch normalization, conv 2d and relu.

# linear_1d(sz)
Does 1d linear interpolation (curve fitting using linear polynomials to construct new data points within the range).

# linear_2d(sz)
2D linear interpolation kernel

# transpose_upsample2d(x, factor, constant=True)
2D upsampling (inserting zero-valued samples between original samples to increase the sampling rate) operator using transposed convolution transposed convolutions, padding has the reverse effect and it decreases the size of the output (in transposed convolutions, padding has the reverse effect from normal convolutions - it decreases the size of the output).

# build_FCN(image, n_class, n_level, n_filter, n_block, training, same_dim=32, fc=64)
Build a fully convolutional network for segmenting an input image into n_class classes and return the logits map.

# build_ResNet(image, n_class, n_level, n_filter, n_block, training, use_bottleneck=False, same_dim=32, fc=64)
Build a fully convolutional network with residual learning units for segmenting an input image into n_class classes and return the logits map.

### Method flow
Lots of loose functions called by build_FCN to construct the model. The file is called by train_network.py to build the network dynamically.

###############################################

# train_network.py
## File dependencies
From network.py uses build_FCN to build the model layers.

## Deprecations
?

## Libraries
import os
import time
import random
import numpy as np
import nibabel as nib
import tensorflow.compat.v1 as tf
from network import build_FCN

## Deployment parammeters
seq_name = 'sa'
image_size = 192 (after crop)
train_batch_size = 2
train_iteration = 10
num_filter = 16
num_level = 5 (network levels)
learning_rate = 1e-3
dataset_dir = 'data'
log_dir = log
checkpoint_dir = checkpoint
model_path = './trained_model'

## Functions
# tf_categorical_accuracy(pred, truth)
Computes the mean of elements across dimensions of a tensor as an accuracy metric for the model

# tf_categorical_dice(pred, truth, k)
Caculated dice overlap metric for some label k.

# crop_image(image, cx, cy, size)
Crop a 3D image using a bounding box centred at (cx, cy) with specified size.

# rescale_intensity(image, thres=(1.0, 99.0))
Rescale the image intensity to the range of [0, 1]

# data_augmenter(image, label, shift, rotate, scale, intensity, flip)
Online data augmentation.
Perform affine (transformation that preserves collinearity (i.e., all points lying on a line initially still lie on a line after transformation) and ratios of distances (e.g., the midpoint of a line segment remains the midpoint after transformation)) transformation on image and label,
which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).

# get_random_batch(filename_list, batch_size, image_size=192, data_augmentation=False, shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False)
Randomly select batch_size images from filename_list. It reads the image and label, normalises the image size, rescales the intensity, appends the image slices to the batch using a list. Converts images to a numpy array, adds a channel dimention and perform data augmentation.

# controller()
This is functionally the main method. It goes through each subset of data directories (training, validation, and test), checks the existence of the image and label map at ED and ES time frames and adds their name to a list. Prepares tensors for image and label map pairing. Prints out the placeholders name. 

### Method flow
Controller is called. Iterate through data directories and ED / ES data images. Create tf placeholder (_pl) of the data image and label to later feed them to the graph. Create training place holder. Set number of class to 4 since using short axis. Here we see the distributiveness of the model... Number of filters of the model is being set in the training file - it should be set in the model file (network.py). Call build_FCN from network.py to build the network that outputs the logits (he vector of raw (non-normalized) predictions that a classification model generates) which are then fed to a softmax function to calculate probabilities and predictions. Loss and evaluation metrics are then deployed. Operators associated with batch_normalization are then added  (adam optimizer which is an algorithm that leverages the power of adaptive learning rates methods to find individual learning rates for each parameter). Start training. Create tf session, summary writer, initialise variables, create saver, start training iteration. Get random batch of images/labels. Call sess.run to do stochastic optimisation using the batch - return train loss and train accuracy. After 10 iterations of training perform validation. Output all the summary values regarding training accuracy and loss in segmenting left ventricle, myocardium and right ventricle. Print the iteration and these results.