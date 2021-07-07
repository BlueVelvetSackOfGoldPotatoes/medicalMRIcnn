import matplotlib.image as mpimg
from matplotlib import pyplot as plt

img1 = mpimg.imread('/home/goncalo/Documents/thesisCode_bitbucket/itc-main-repo/Goncalo/goncalo/data/data_jpg/sa/output-frame000-slice000.jpg')

def show_img(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.savefig('myfig.png') # To save the figure

def check_dimension(img):
    print(img.shape)

def main():
    show_img(img1)
    check_dimension(img1)

if __name__== '__main__':
    main()
