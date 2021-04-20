#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy


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
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right','Above','Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub','Below']:
                    return_string += ['_','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup','Above']:
                    return_string += ['^','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string

def main(args):
    gtd_root_path = '../data/{}/'.format(args.dataset_type)
    latex_root_path = '../data/{}/'.format(args.dataset_type)

    gtd_paths = ['caption/train_caption', 'caption/test_caption']
    for gtd_path in gtd_paths:
        gtd_files = os.listdir(gtd_root_path + gtd_path + '/')
        f_out = open(latex_root_path + gtd_path + '.txt', 'w')
        for process_num, gtd_file in enumerate(gtd_files):
            # gtd_file = '510_em_101.gtd'
            key = gtd_file[:-4] # remove .gtd
            f_out.write(key + '\t')
            gtd_list = []
            gtd_list.append(['<s>',0,-1,'root'])
            with open(gtd_root_path + gtd_path + '/' + gtd_file) as f:
                lines = f.readlines()
                for line in lines[:-1]:
                    parts = line.split()
                    sym = parts[0]
                    childid = int(parts[1])
                    parentid = int(parts[3])
                    relation = parts[4]
                    gtd_list.append([sym,childid,parentid,relation])
            latex_list = convert(1, gtd_list)
            if 'illegal' in latex_list:
                print (key + ' has error')
                latex_string = ' '
            else:
                latex_string = ' '.join(latex_list)
            f_out.write(latex_string + '\n')
            # sys.exit()

        if (process_num+1) // 2000 == (process_num+1) * 1.0 / 2000:
            print ('process files', process_num)

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'CROHME' # CROHME / 20K / MATHFLAT(TODO)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))