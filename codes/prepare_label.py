#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy


def gen_gtd_label(args):
    gtd_root_path = '../data/{}/caption/'.format(args.dataset_type)
    bfs1_path = '../data/{}/caption/'.format(args.dataset_type)
    gtd_paths = ['train_caption', 'test_caption']

    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_label_gtd.pkl'
        out_label_fp = open(outpkl_label_file, 'wb')
        label_lines = {}
        process_num = 0
        
        file_list  = os.listdir(gtd_root_path + gtd_path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .gtd
            if key in ['fa66375ede8be1c192a1acc2bc62b575.jpg']:
                continue
            with open(gtd_root_path + gtd_path + '/' + file_name) as f:
                lines = f.readlines()
                label_strs = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        sym = parts[0]
                        align = parts[1]
                        related_sym = parts[2]
                        realign = parts[3]
                        relation = parts[4]
                        string = sym + '\t' + align + '\t' + related_sym + '\t' + realign + '\t' + relation
                        label_strs.append(string)
                    else:
                        print ('illegal line', key)
                        sys.exit()
                label_lines[key] = label_strs

            process_num = process_num + 1
            if process_num // 2000 == process_num * 1.0 / 2000:
                print ('process files', process_num)

        print ('process files number ', process_num)

        pkl.dump(label_lines, out_label_fp)
        print ('save file done')
        out_label_fp.close()


def gen_gtd_align(args):
    gtd_root_path = '../data/{}/caption/'.format(args.dataset_type)
    bfs1_path = '../data/{}/caption/'.format(args.dataset_type)
    gtd_paths = ['train_caption', 'test_caption']
    
    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_label_align_gtd.pkl'
        out_label_fp = open(outpkl_label_file, 'wb')
        label_aligns = {}
        process_num = 0
    
        file_list  = os.listdir(gtd_root_path + gtd_path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .gtd
            if key in ['fa66375ede8be1c192a1acc2bc62b575.jpg']:
                continue
            with open(gtd_root_path + gtd_path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = len(lines)
                align = numpy.zeros([wordNum, wordNum], dtype='int8')
                wordindex = -1

                for line in lines:
                    wordindex += 1
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        realign = parts[3]
                        realign_index = int(realign)
                        align[realign_index,wordindex] = 1
                    else:
                        print ('illegal line', key)
                        sys.exit()
                label_aligns[key] = align

            process_num = process_num + 1
            if process_num // 2000 == process_num * 1.0 / 2000:
                print ('process files', process_num)

        print ('process files number ', process_num)

        pkl.dump(label_aligns, out_label_fp)
        print ('save file done')
        out_label_fp.close()

def main(args):
    gen_gtd_label(args)
    gen_gtd_align(args)
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'CROHME' # CROHME / 20K / MATHFLAT(TODO)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))