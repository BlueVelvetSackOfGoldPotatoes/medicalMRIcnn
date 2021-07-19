import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2

def detect_blur_fft(image, size=60, thresh=10):
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	# check to see if we are visualizing our output
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)

	return (mean, mean <= thresh)

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input directory of images")
    ap.add_argument("-t", "--thresh", type=int, default=29,
        help="threshold for our blur detector to fire")
    args = vars(ap.parse_args())

    for imagePath in paths.list_images(args["images"]):
        orig = cv2.imread(imagePath)
        orig = imutils.resize(orig, width=250)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # apply blur detector using the FFT
        (mean, blurry) = detect_blur_fft(gray, size=60,
            thresh=args["thresh"])

        image = np.dstack([orig] * 3)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
        text = text.format(mean)
        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2)
        print("[INFO] {}".format(text))

        cv2.imshow("Output", image)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
