import numpy as np
from shutil import copyfile


import os
#

sample = open('data/sample_submission.csv','r')

lab = []
for index,item in enumerate(sample.readlines()):

    if index ==0:
        lab = [x.strip() for x in item.split(',')[1:]]
sample.close()

#build id dic

label_dict = dict()
for index,item in enumerate(lab):
    label_dict[item] = index

# print (label_dict)
# 'Quercus_Crassifolia': 60, ....


#
#
# train_file = open("data/train.csv",'r')
#
# ids = []
# labels = []
# features = []
# for line in train_file.readlines():
#
#     seq = line.split(',')
#     if seq[1] != 'species':
#         ids.append(seq[0])
#         labels.append(seq[1].strip())
#         features.append(seq[2:])
# train_file.close()
#
# print (labels)
#
# transfer_file = open('data/ctrain.csv','w')
#
# for index,item in enumerate(ids):
#     transfer_file.write(','.join([x.strip() for x in features[index]]))
#     label = label_dict[labels[index]]
#     transfer_file.write(',')
#     transfer_file.write(str(label))
#     transfer_file.write('\n')
# transfer_file.close()
#

# Gather images: these are ids in each
def load_ids():
    train = []
    test = []
    train_tag = []

    for index,line in enumerate(open('data/train.csv','r').readlines()):
        if index > 0:
            train.append(line.split(',')[0])
            train_tag.append(line.split(',')[1])

    for index,line in enumerate(open('data/test.csv','r').readlines()):
        if index > 0:
            test.append(line.split(',')[0])
    print (train_tag)
    return train,test,train_tag

def name_to_id(train,dic,name_list):
    id = []
    for index,item in enumerate(train):
        id.append(dic[name_list[index]])
    name2tag = dict()
    for index,item in enumerate(train):
        name2tag[item] = id[index]
    print (name2tag)
    return name2tag

train,test,train_tag= load_ids()
print (train)
train_tag_id = name_to_id(train,label_dict,train_tag)

# split to image folders
def split_images(train,test,train_tag_id):
    dir = 'data/imgs/'
    target_dir = 'data/split/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(target_dir+'/train/')
        os.makedirs(target_dir + '/test/')

    all_images = os.listdir(dir)
    # print (all_images)
    for img in all_images:
        if img.endswith('.jpg'):
            img_id = img.split('.')[0]

            if img_id in train:
                tag_id = train_tag_id[img_id]
                if not os.path.exists(target_dir+'/train/'+str(tag_id)+'/'):
                    os.makedirs(target_dir+'/train/'+str(tag_id)+'/')
                copyfile(dir+img,target_dir+'/train/'+str(tag_id)+'/'+img)
            else:

                copyfile(dir+img, target_dir + '/test/' + img)

    print ('Finish')

split_images(train,test,train_tag_id)





