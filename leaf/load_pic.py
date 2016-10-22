from scipy import misc
import numpy as np
import os,sys

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




# ext = np.expand_dims(test,axis=2)



# re-scale pics
from PIL import Image
from resizeimage import resizeimage

dir = 'data/img_convert/'
dir_new = 'data/imgs/'


def rescale(dir,dir_new,size):
    for img in os.listdir(dir):
        if img.endswith('.jpg'):
            with open(dir+img, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_contain(image, [size, size])
                    cover.save(dir_new+img, cover.format)

    print ('Finished')



def load_train_pics(dir='data/split/train/',num_class = 99):
    # actual size:
    # labels :(990, 99)
    # data: (990, 600, 600, 1)
    x = []
    y = []
    for label in os.listdir(dir):
        for img in os.listdir(dir+'/'+label):
            if img.endswith('.jpg'):
                test = misc.imread(dir+'/'+label + '/' + img,mode='L')
                x.append(test)
            y.append(int(label))
    size = len(y)
    expanded_y = np.reshape(y,newshape=(size,1))
    one_hot_y = []
    for item in expanded_y:
        one_hot_y.append(np.eye(num_class)[item])
    reshape_y = np.reshape(one_hot_y,newshape=(size,num_class))

    expanded_x = np.expand_dims(x, axis = 3)
    print ("Loading Training x....",np.shape(expanded_x))
    print ("Loading Training y....",np.shape(reshape_y))

    return expanded_x,reshape_y

def load_test_pics(dir='data/split/test/',num_class = 99):
    # actual size:
    # data: (549, 600, 600, 1)
    x = []
    for img in os.listdir(dir):
        if img.endswith('.jpg'):
            test = misc.imread(dir+'/'+img,mode='L')
            x.append(test)
    expanded_x = np.expand_dims(x, axis = 3)
    print ("Loading Testing x....",np.shape(expanded_x))
    return expanded_x


def load_batch(x,y,size):
    # randomly select from x and y
    ids = np.random.choice(x.shape[0], size)
    batch_x = x[ids]
    batch_y = y[ids]
    return batch_x,batch_y
