#!/usr/bin/env python
import argparse
import os
import sys
import _pickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave
from common.general_utils import folder_exists, concat_text_files


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main(args):
    op_modes = args.op_mode.split('_')
    txt_fnames = []
    for op_mode in op_modes:
        lower_op_mode = op_mode.lower()
        image_path = './{}/image/off_image_{}/'.format(args.dataset_type, lower_op_mode)
        caption_path = os.path.join(_this_folder_, args.dataset_type, 'caption', lower_op_mode+'_caption.txt')
        txt_fnames.append(caption_path)

        scpFile = open(caption_path)
        outFile = os.path.join(_this_folder_, args.dataset_type, 'offline-{}.pkl'.format(lower_op_mode))
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

                im = imread(image_file)
                channels = 1 if len(im.shape) == 2 else 3
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
                    print('process sentences ', sentNum)

        print('load images done. sentence number ', sentNum)

        pkl.dump(features, oupFp_feature)
        print('Op_mode : {}, save file done'.format(args.op_mode))
        oupFp_feature.close()

    if args.op_mode == 'TRAIN_TEST':
        rst_fname = os.path.join(_this_folder_, args.dataset_type, 'caption', 'total_caption.txt')
        concat_ = concat_text_files(txt_fnames=txt_fnames, rst_fname=rst_fname)
        if concat_:
            print('Generated total_caption.txt')
    
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")
    parser.add_argument("--op_mode", required=True, choices=['TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    
    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'CROHME' # CROHME / 20K / MATHFLAT(TODO)
OP_MODE = 'TRAIN_TEST'


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))
            