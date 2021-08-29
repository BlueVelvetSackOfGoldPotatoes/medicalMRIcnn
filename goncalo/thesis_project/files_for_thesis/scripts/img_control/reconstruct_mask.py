from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
import os
import numpy as np

path_sunnybroke_contours = "/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/sunnybroke_dataset/manualcontours/SCD_ManualContours"

def create_img_vector(file_path):
    # image_array = np.empty((0,2),dtype=np.uint8)
    image_array = []
    with open(file_path) as file:
        lines = file.readlines()
        counter = 1
        for line in lines:
            # We've caught the pair
            line = line.strip()
            x = float(line.split(' ')[0])
            y = float(line.split(' ')[1])
            coordinate = [x, y]
            image_array.append(coordinate)
        image_array = np.array(image_array, dtype=np.uint8)
    return image_array

def build_and_save_contour(file_path, name):
    coords = create_img_vector(file_path)
    img = Image.new("RGB", (192,192))
    draw = ImageDraw.Draw(img)
    dotSize = 2

    for (x,y) in coords:
        draw.rectangle([x,y,x+dotSize-1,y+dotSize-1], fill="white")
    # img.show()
    img.save(name + ".jpg")
    print("Image {} DONE! -------------------------".format(name))

def main():
    # crawl the files in one mask path, first do with one mask the above function
    # for mask in file_path:
            
    for root, dirnames, filenames in os.walk(path_sunnybroke_contours):
        for filename in filenames:
            name, type = os.path.splitext(filename)
            if type == ".txt":
                path_name=""
                # for path_n in os.path.join(root, "").split('/')[:-3]:
                for path_n in os.path.join(root, "").split('/')[1:-1]:
                    path_name = path_name + "/" + path_n
                new_file_path = path_name + "/" + name
                build_and_save_contour(os.path.join(root, filename), new_file_path)

if __name__ == '__main__':
    main()