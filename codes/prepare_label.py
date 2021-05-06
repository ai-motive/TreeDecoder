#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy


def gen_gtd_label(args):
    gtd_root_path = args.gtd_path
    bfs1_path = args.label_pkl_path

    outpkl_label_file = bfs1_path
    out_label_fp = open(outpkl_label_file, 'wb')
    label_lines = {}
    process_num = 0

    file_list  = os.listdir(gtd_root_path)
    for file_name in file_list:
        key = file_name[:-4] # remove suffix .gtd
        if key in ['fa66375ede8be1c192a1acc2bc62b575.jpg']:
            continue
        with open(gtd_root_path + '/' + file_name) as f:
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
    gtd_root_path = args.gtd_path
    bfs1_path = args.align_pkl_path

    outpkl_label_file = bfs1_path
    out_label_fp = open(outpkl_label_file, 'wb')
    label_aligns = {}
    process_num = 0

    file_list  = os.listdir(gtd_root_path)
    for file_name in file_list:
        key = file_name[:-4] # remove suffix .gtd
        if key in ['fa66375ede8be1c192a1acc2bc62b575.jpg']:
            continue
        with open(gtd_root_path + '/' + file_name) as f:
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
    parser.add_argument("--op_mode", required=True, choices=['TRAIN', 'TEST'], help="operation mode")
    parser.add_argument("--gtd_path", required=True, help="Root gtd path")
    parser.add_argument("--label_pkl_path", required=True, help="Root label pickle path")
    parser.add_argument("--align_pkl_path", required=True, help="Root align pickle path")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'MATHFLAT' # CROHME / 20K / MATHFLAT
OP_MODE = 'TRAIN'

if __name__ == '__main__':
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))