#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy
from latex2gtd import ABOVE_SYMBOLS, BELOW_SYMBOLS, ARRAY_SYMBOLS, LATEX_SYMBOLS, split_string_to_latex_symbols


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if (gtd_list[nodeid][0] in ABOVE_SYMBOLS) or (gtd_list[nodeid][0] in BELOW_SYMBOLS):
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re == 'Above':
                    return_string += ['{'] + convert(idx, gtd_list) + ['}']
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re == 'Below':
                    return_string += ['{'] + convert(idx, gtd_list) + ['}']
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re == 'Right':
                    return_string += convert(idx, gtd_list)
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re not in ['Right','Above','Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re == 'Inside':
                    return_string += ['{'] + convert(idx, gtd_list) + ['}']
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re in ['Sub','Below']:
                    return_string += ['_','{'] + convert(idx, gtd_list) + ['}']
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re in ['Sup','Above']:
                    return_string += ['^','{'] + convert(idx, gtd_list) + ['}']
            for i in range(len(child_list)):
                idx, re = child_list[i][1], child_list[i][2]
                if re in ['Right']:
                    return_string += convert(idx, gtd_list)

        return return_string

def main(args):
    print('Dataset type : {}'.format(args.dataset_type))
    gtd_root_path = args.gtd_root_path
    latex_root_path = args.latex_root_path

    gtd_paths = ['gtd']
    for gtd_path in gtd_paths:
        gtd_files = os.listdir(gtd_root_path + gtd_path + '/')
        gtd_files = sorted(gtd_files)
        f_out = open(latex_root_path + gtd_path + '.txt', 'w')
        for process_num, gtd_file in enumerate(gtd_files):
            # gtd_file = '510_em_101.gtd'
            key = gtd_file[:-4] # remove .gtd
            f_out.write(key + '\t')
            gtd_list = []
            gtd_list.append(['<s>',0,-1,'root'])

            # if '7uo72ow1frx686al_p_crop_007' not in key:
            #     continue

            with open(gtd_root_path + gtd_path + '/' + gtd_file) as f:
                lines = f.readlines()
                for line in lines[:-1]:
                    parts = line.split()
                    sym = parts[0]
                    childid = int(parts[1])
                    parentid = int(parts[3])
                    relation = parts[4]
                    gtd_list.append([sym, childid, parentid, relation])

            latex_list = convert(1, gtd_list)

            if 'illegal' in latex_list:
                print (key + ' has error')
                latex_string = ' '
            else:
                latex_string = ' '.join(latex_list)

            if args.dataset_type == 'CROHME':
                out_string = latex_string + '\n'
            elif args.dataset_type == 'MATHFLAT':
                strip_latex = latex_string

                if strip_latex[:2] == '{}':
                    strip_latex = strip_latex[2:]

                latex_parts = split_string_to_latex_symbols(strip_latex,
                                                      latex_symbols=ARRAY_SYMBOLS + LATEX_SYMBOLS)
                out_string = "".join(latex_parts)

            f_out.write(out_string + '\n')

            print(" [GENERATE_LATEX] # Processing {} ({:d}/{:d}) : {}".format(key, (process_num + 1),
                                                                              len(lines), latex_string))

        if (process_num+1) // 2000 == (process_num+1) * 1.0 / 2000:
            print ('process files', process_num)

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")
    parser.add_argument("--tgt_mode", required=True, choices=['TRAIN', 'TEST'], help="Target mode")
    parser.add_argument("--gtd_root_path", default="../data/CROHME/", help="Root path of gtd files")
    parser.add_argument("--latex_root_path", default="../data/CROHME/", help="Root path of latex files")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'MATHFLAT' # CROHME / 20K / MATHFLAT
TGT_MODE = 'TRAIN' # TRAIN / TEST


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--tgt_mode", TGT_MODE])
            sys.argv.extend(["--gtd_root_path", "/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/20000_29999/train/tree_math_gt/"])
            sys.argv.extend(["--latex_root_path", "/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/20000_29999/train/tree_math_gt/latex/"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))