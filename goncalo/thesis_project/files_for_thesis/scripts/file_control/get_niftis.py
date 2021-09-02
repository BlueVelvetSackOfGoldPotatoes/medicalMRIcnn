# Used to translate umcg jpg frames into nifti sequence

import os
import glob
import numpy as np
import ntpath
import nibabel as nib
import cv2
from PIL import Image
from numpy import asarray

def main():
	# deconstruct images in numpy array and concatenate a 4d array (288, 288, 2, 8)
	data_dir = '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data/ukbb/ukbb_original_png'

	frames = np.empty((512, 512, 1, 13))
	for file in glob.glob(os.path.join(data_dir,"*.png")):
		file_name_comp = ntpath.basename(file)
		# file_name = file_name_comp[14:-13]
		frame_n = int(file_name_comp[-5])
		dezena = file_name_comp[-6]
		if dezena != "_":
			dezena = 10
			frame_n = frame_n + 10
		# print(frame_n)
		# ED
		# if file_name == "0":
		image = cv2.imread(file, 0)
		img = asarray(image)
		print(img.shape)
		frames[:,:,0,frame_n] = img
		# ES
		# if file_name == "1":
		# 	image = Image.open(file)
		# 	img = asarray(image)
		# 	frames[:,:,1,frame_n] = img
	print(frames.shape)
	img = nib.Nifti1Image(frames, np.eye(4))
	nib.save(img, '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data/ukbb/ukbb_original_png/sa.nii.gz')

if __name__ == '__main__':
    main()









































































	# data_dir = '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/data_jpg_demo/4/jpg'
	# data_list = sorted(os.listdir(data_dir))
	# file_names = glob.glob(data_dir + '/*.jpg')

	# reader = sitk.ImageSeriesReader()
	# reader.SetFileNames(file)
	# vol = reader.Execute()

	# sitk.WriteImage(vol, 'sa.nii.gz')