#!/usr/bin/env python

import os
import sys
import _pickle as pkl
# import cPickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave

op_mode = 'test'    # <---convert

if op_mode == 'train':
    image_path = './off_image_train/'
    scpFile = open('train_caption.txt')
    outFile = 'offline-train.pkl'
elif op_mode == 'test':
    image_path='./off_image_test/' # for test.pkl
    scpFile = open('test_caption.txt')
    outFile='offline-test.pkl'

oupFp_feature=open(outFile,'wb')

features={}
channels=1
sentNum=0

while 1:
    line=scpFile.readline().strip() # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '_' + str(0) + '.bmp'
        im = imread(image_file)
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            image_file = image_path + key + '_' + str(channel) + '.bmp'
            im = imread(image_file)
            mat[channel,:,:] = im
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum / 500 == sentNum * 1.0 / 500:
            print('process sentences ', sentNum)

print('load images done. sentence number ',sentNum)

pkl.dump(features,oupFp_feature)
print('save file done')
oupFp_feature.close()
