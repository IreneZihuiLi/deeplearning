from scipy import misc
import numpy as np
import os,sys

# dir = 'data/images/'
# width = []
# height = []
# for img in os.listdir(dir):
#     if img.endswith('.jpg'):
#         test = misc.imread(dir+img)
#         w,h = np.shape(test)
#         width.append(w)
#         height.append(h)
#
# print (np.average(np.asarray(width)))
# print (np.average(np.asarray(height)))

# 493, 693


# ext = np.expand_dims(test,axis=2)

# resize test

#
# dir = 'data/images/'
# dir_conv = 'data/img_convert/'
# for img in os.listdir(dir):
#     if img.endswith('.jpg'):
#         test = misc.imread(dir+img)
#         x2 = np.full(np.shape(test),255,dtype=np.int)
#         latest = np.abs(np.subtract(test, x2))
#         misc.imsave(dir_conv+img,latest)




# re-scale pics
#

from PIL import Image
from resizeimage import resizeimage
dir = 'data/img_convert/'
dir_new = 'data/imgs/'

for img in os.listdir(dir):
    if img.endswith('.jpg'):
        with open(dir+img, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_contain(image, [500, 700])
                cover.save(dir_new+img, cover.format)

print ('Finished')