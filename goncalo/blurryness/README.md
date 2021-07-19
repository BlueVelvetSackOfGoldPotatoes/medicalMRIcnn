# DETECTING BLURRYNESS USING LAPLACIAN

Using the variation of the Laplacian by Pech-Pacheco et al. in their 2000 ICPR paper, Diatom autofocusing in brightfield microscopy: a comparative study.

## HOW DOES IT WORK?

### LAPLACIAN
Take a single channel of an image (presumably grayscale) and convolve it with the following 3 x 3 kernel, and then take the variance (i.e. standard deviation squared) of the response.

If the variance falls below a pre-defined threshold, then the image is considered blurry; otherwise, the image is not blurry. 
The Laplacian operator measures the 2nd derivative of an image highlighting regions of an image containing rapid intensity changes. It works for edge detection as well. Then, if an image contains high variance there is a wide spread of responses: hard well-defined edges which indicate an in-focus image. But if there is very low variance, this indicates a low spread of responses, and thus no well defined edges in the image. The more an image is blurred, the less edges there are.

Setting the correct threshold which can be quite domain dependent. Too low of a threshold and the algorithm incorrectly marks images as blurry when they are not. Too high of a threshold then images that are actually blurry will not be marked as blurry. I think for MRI images the threshold is going to be quite sensitive.

### FAST FOURIER TRANSFORM
Liu et al.’s 2008 CVPR publication, Image Partial Blur Detection and Classification.
The Fourier Transform can be used to decompose an image into its sine and cosine components. The output of the transformation represents the image in the Fourier or frequency domain, while the input image is the spatial domain equivalent. In the Fourier domain image, each point represents a particular frequency contained in the spatial domain image. 

Again, the higher the variance in frequency the more blur. 

### RUN THE SCRIPT USING

 for detect_blurry.py:
    python detect_blurry.py --images images --thresh 100
 for detect_blurry2.py
    python detect_blurry2.py --image images/resume_01.png --thresh 27
    

### PACKAGES
pip install imutils