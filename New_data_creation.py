from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import cv2
import time
import cmapy


final_cmap_list = [ 'CMRmap_r' ,'Paired' , 'Paired_r' , 'YlGnBu', 'afmhot',  'gist_stern', 'gist_rainbow' , 'gist_stern_r' , 'hsv' , 'jet_r' , 'nipy_spectral_r'  , 'ocean' ]
print(len(final_cmap_list))




start_time = time.time()
def look_sample(img_filter , img_list):
    kernel = np.array([[0,-1,0], [-1, 5,-1],[0,-1,0]])
    b = 0

    for img_direc in img_list:
        batch_start_time = time.time()
        img = Image.open(img_direc)

        basewidth = 512

        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)

        img = np.array(img)

        img = ndimage.median_filter(img, 3)    #Median filter
        img = cv2.filter2D(img, -1, kernel)    #sharpening conv filter
        #img = gaussian_filter(img, sigma=2)   #Gaussian_Filter
        img = img * 255

        img = cv2.applyColorMap(img, cmapy.cmap(img_filter))
        img = ndimage.median_filter(img, 3)    #Median filter
        img_final = Image.fromarray(img)
        img_final.save(img_direc)
        if b % 100 == 0:
            print(b , 'image_saved. This image took' , time.time() - batch_start_time , 'sec' )
        b += 1

    print('Total' , b , 'images saved in ' ,   time.time() - start_time)

img_list = glob("/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classific"
                "ation-master/Modified_data/**/*.jpg" , recursive=True)


#print(img_list)

print(len(img_list))

img_filter = 'gist_rainbow'
look_sample(img_filter , img_list)
