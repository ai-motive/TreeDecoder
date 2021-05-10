#!/usr/bin/env python
import argparse
import os
import sys
import _pickle as pkl
import numpy
import cv2
from scipy.misc import imread, imresize, imsave
from utility.general_utils import folder_exists, file_exists, concat_text_files


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main(args):
    op_mode = args.op_mode
    caption_path = args.cptn_path
    image_path = args.crop_path
    image_pkl_path = args.img_pkl_path

    scpFile = open(caption_path)
    outFile = image_pkl_path
    oupFp_feature = open(outFile, 'wb')

    features = {}
    sentNum = 0

    while 1:
        line = scpFile.readline().strip()  # remove the '\r\n'
        if not line:
            break
        else:
            key = line.split('\t')[0]
            if args.dataset_type == 'CROHME':
                ext = '.bmp'
                image_file = image_path + key + '_' + str(0) + ext
            elif args.dataset_type == '20K':
                ext = '.png'
                image_file = image_path + key + ext
            elif args.dataset_type == 'MATHFLAT':
                ext = '.jpg'
                image_file = image_path + key + ext

            image_file_ = file_exists(image_file)
            if not(image_file_):
                continue
            im = imread(image_file)

            if len(im.shape) == 2:
                channels = 1
            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
                channels = 1

            mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
            for channel in range(channels):
                # image_file = image_path + key + '_' + str(channel) + ext
                im = imread(image_file)
                if len(im.shape) == 2:
                    mat[channel, :, :] = im
                else:
                    mat[channel, :, :] = im[:, :, channel]

            sentNum = sentNum + 1
            features[key] = mat
            if sentNum / 500 == sentNum * 1.0 / 500:
                print('process sentences : {}'.format(sentNum))

    print(f'load {op_mode} images done. sentence number ', sentNum)

    pkl.dump(features, oupFp_feature)
    print('Op_mode : {}, save file done : {}'.format(args.op_mode, outFile))
    oupFp_feature.close()

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")
    parser.add_argument("--op_mode", required=True, choices=['TRAIN', 'TEST'], help="operation mode")
    parser.add_argument("--cptn_path", required=True, help="Caption file path")
    parser.add_argument("--crop_path", required=True, help="Crop image folder path")
    parser.add_argument("--img_pkl_path", required=True, help="Crop image pickle path")
    
    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'MATHFLAT' # CROHME / 20K / MATHFLAT
OP_MODE = 'TRAIN'


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))
            