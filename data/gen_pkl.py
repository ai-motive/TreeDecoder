#!/usr/bin/env python

import os
import sys
import _pickle as pkl
# import cPickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave

dataset_type = '20K'    # CHROHME / 20K
op_mode = 'test'    # train / test

if op_mode == 'train':
    image_path = './{}/off_image_train/'.format(dataset_type)
    scpFile = open(os.path.join(dataset_type, 'train_caption.txt'))
    outFile = os.path.join(dataset_type, 'offline-train.pkl')
elif op_mode == 'test':
    image_path='./{}/off_image_test/'.format(dataset_type) # for test.pkl
    scpFile = open(os.path.join(dataset_type, 'test_caption.txt'))
    outFile = os.path.join(dataset_type, 'offline-test.pkl')

oupFp_feature = open(outFile,'wb')

features = {}
sentNum = 0

while 1:
    line = scpFile.readline().strip() # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        if dataset_type == 'CHROHME':
            ext = '.bmp'
            image_file = image_path + key + '_' + str(0) + ext

        elif dataset_type == '20K':
            ext = '.png'
            image_file = image_path + key + ext

        im = imread(image_file)
        channels = 1 if len(im.shape) == 2 else 3
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            # image_file = image_path + key + '_' + str(channel) + ext
            im = imread(image_file)
            if len(im.shape) == 2:
                mat[channel,:,:] = im
            else:
                mat[channel,:,:] = im[:,:,channel]

        sentNum = sentNum + 1
        features[key] = mat
        if sentNum / 500 == sentNum * 1.0 / 500:
            print('process sentences ', sentNum)

print('load images done. sentence number ',sentNum)

pkl.dump(features,oupFp_feature)
print('save file done')
oupFp_feature.close()
