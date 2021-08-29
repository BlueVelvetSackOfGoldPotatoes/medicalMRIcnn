def normalize_filled(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im, cnt, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # fill shape
    cv2.fillPoly(img, pts=cnt, color=(255,255,255))
    bounding_rect = cv2.boundingRect(cnt[0])
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
    img_resized = cv2.resize(img_cropped_bounding_rect, (300, 300))
    return img_resized

imgs = [imgQuery, imgHDMI, imgDVI, img5PinDin, imgDB25]
imgs = [normalize_filled(i) for i in imgs]

for i in range(1, 6):
    plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
    print(cv2.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))