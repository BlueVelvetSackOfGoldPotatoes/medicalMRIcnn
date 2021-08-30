import numpy as np
import cv2
import matplotlib.pyplot as plt
import shapely.geometry as shapgeo
from scipy.spatial.distance import directed_hausdorff
from skimage import metrics

def calc_hausdorff(true, pred):
  return metrics.hausdorff_distance(true, pred)

def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

def center_image(image):
  height, width = image.shape
  wi=(width/2)
  he=(height/2)

  ret,thresh = cv2.threshold(image,95,255,0)

  M = cv2.moments(thresh)

  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])

  offsetX = (wi-cX)
  offsetY = (he-cY)
  T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
  centered_image = cv2.warpAffine(image, T, (width, height))

  return centered_image

def standardize(img, resize=False):
    if resize:
      resized = cv2.resize(img, (162, 208))
    img_grey = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    thresh = 1
    img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

    final_img = center_image(img_binary)

    # _, recolored = cv2.threshold(resized, 1, 255, cv2.THRESH_BINARY)
    print(final_img.shape)
    plt.imshow(final_img)
    plt.show()
    return final_img

def calc_area(true, pred):
    contours_true, _ = cv2.findContours(true.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_pred, _ = cv2.findContours(pred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    M_true = cv2.moments(contours_true[0])
    M_pred = cv2.moments(contours_pred[0])

    if M_true['m00'] > M_pred['m00']:
        # Find (approximated) contours of inner and outer shape
        outer = [cv2.approxPolyDP(contours_true[0], 0.1, True)]
        inner = [cv2.approxPolyDP(contours_pred[0], 0.1, True)]
    else:
        outer = [cv2.approxPolyDP(contours_pred[0], 0.1, True)]
        inner = [cv2.approxPolyDP(contours_true[0], 0.1, True)]

    # Images have the same shape
    h, w = true.shape[:2]
    outer = np.vstack(outer).squeeze()
    inner = np.vstack(inner).squeeze()

    # Calculate centroid of inner contour
    M = cv2.moments(inner)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Calculate maximum needed radius for later line intersections
    r_max = np.min([cx, w - cx, cy, h - cy])

    # Set up angles (in degrees)
    angles = np.arange(0, 360, 4)

    # Initialize distances
    dists = np.zeros_like(angles)

    # Prepare calculating the intersections using Shapely
    poly_outer = shapgeo.asLineString(outer)
    poly_inner = shapgeo.asLineString(inner)

    # Iterate angles and calculate distances between inner and outer shape
    for i, angle in enumerate(angles-1):

        # Convert angle from degrees to radians
        angle = angle / 180 * np.pi

        # Calculate end points of line from centroid in angle's direction
        x = np.cos(angle) * r_max + cx
        y = np.sin(angle) * r_max + cy
        points = [(cx, cy), (x, y)]

        # Calculate intersections using Shapely
        poly_line = shapgeo.LineString(points)
        insec_outer = np.array(poly_outer.intersection(poly_line))
        insec_inner = np.array(poly_inner.intersection(poly_line))

        # Calculate distance between intersections using L2 norm
        dists[i] = np.linalg.norm(insec_outer - insec_inner)

    running_sum = 0
    for distance in dists:
      running_sum = distance + running_sum
    return running_sum / len(dists)

def main():
    # print(standardize("/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/demo_image/3/seg_matrice_imgs/seg_matrice_img_1_8.png"))
    true = standardize("/home/goncalo/Downloads/LV/A036/036_Ileen_RN_cine.con-sl02-fr033-endo.png")
    pred = standardize("/home/goncalo/Downloads/LV/A036/036_Ileen_RN_cine.con-sl03-fr033-epi.png")
    print('Hausdorff distance: {}'.format(calc_hausdorff(true, pred)))
    print('Dice coefficient:{}'.format(compute_dice_coefficient(true, pred)))
    print('Mean contour distance: {}'.format(calc_area(true,pred)))

if __name__ == '__main__':
    main()