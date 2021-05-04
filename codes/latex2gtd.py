#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy
import re


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

BELOW_ABOVE_SYMBOLS = ['\\sum', '\\int', '\\lim']
ABOVE_SYMBOLS = ['\\frac', '\\overset']
BELOW_SYMBOLS = ['\\underset']
INSIDE_SYMBOLS = ['\\boxed', '\\dot', '\\hat',
                  '\\overbrace', '\\overleftarrow', '\\overline', '\\prod', '\\sqrt',
                  '\\text', '\\textcircled', '\\underbrace', '\\undergroup', '\\underline',
                  '\\vec']

BEGIN_ARRAY_SYMBOLS = ['\\begin{array}{l}', '\\begin{pmatrix}', '\\begin{vmatrix}', '\\begin{Vmatrix}']
END_ARRAY_SYMBOLS = ['\\end{array}', '\\end{pmatrix}', '\\end{vmatrix}', '\\end{Vmatrix}']
ARRAY_SYMBOLS = BEGIN_ARRAY_SYMBOLS + END_ARRAY_SYMBOLS
SCRIPT_SYMBOLS = ['_', '^']
LATEX_SYMBOLS = [
    '\\boxed',
    '\\cos', '\\cot', '\\csc', '\\dot',
    '\\frac', '\\frown', '\\hat', '\\hline',
    '\\int', '\\left',   '\\lim', '\\ln', '\\log',
    '\\max', '\\min', '\\not', '\\overbrace', '\\overleftarrow',
    '\\overline', '\\overset', '\\prod', '\\right', '\\sec',
    '\\sin', '\\smile', '\\space', '\\sqrt', '\\sum',
    '\\tan', '\\text', '\\textcircled', '\\quad', '\\qquad',
    '\\underbrace', '\\undergroup', '\\underline', '\\underset', '\\vec',
    '\\\\', '\\{', '\\}', '\\%', '\\'
]

def split_string_to_latex_symbols(string, latex_symbols=LATEX_SYMBOLS):
    if not(any(latex in string for latex in latex_symbols)):
        return list(string)
    else:
        match_results = []
        for latex in latex_symbols:
            for match in re.finditer(re.escape(latex), string):
                match_results.append((match.start(), match.end(), latex))
                string = string.replace(latex, '*' * len(latex))

        split_string = list(string)
        sort_match_results = sorted(match_results, key=lambda x: (x[0]), reverse=True)

        for match_result in sort_match_results:
            start_idx, end_idx, value = match_result
            split_string[start_idx:end_idx] = [value]

        return split_string


def main(args):
    print('Dataset type : {}'.format(args.dataset_type))
    latex_root_path = args.latex_root_path
    gtd_root_path = args.gtd_root_path

    latex_files = [f'{args.tgt_mode.lower()}_caption.txt']
    for latexF in latex_files:
        latex_file = os.path.join(latex_root_path, latexF)
        gtd_path = os.path.join(gtd_root_path)
        if not os.path.exists(gtd_path):
            os.mkdir(gtd_path)

        with open(latex_file) as f:
            lines = f.readlines()
            for process_num, line in enumerate(lines):
                if '\sqrt [' in line:
                    continue

                if args.dataset_type == 'CROHME':
                    parts = line.split()
                elif args.dataset_type == 'MATHFLAT':
                    core_name = line.replace('\n', '').split('\t')[0]
                    latex_val = line.replace('\n', '').split('\t')[1]
                    strip_latex = latex_val.replace(' ', '')

                    print(" [GENERATE_GTD] # Processing {} ({:d}/{:d}) : {}".format(core_name, (process_num + 1), len(lines), line.strip()))

                    if strip_latex[:2] == '{}':
                        strip_latex = strip_latex[2:]

                    latex_parts = split_string_to_latex_symbols(strip_latex,
                                                                latex_symbols=ARRAY_SYMBOLS+LATEX_SYMBOLS)

                    parts = [core_name] + latex_parts

                if len(parts) < 2:
                    # print('error: invalid latex caption ...', line)   ##
                    continue

                key = parts[0]
                f_out = open(gtd_path + key + '.gtd', 'w')
                raw_cap = parts[1:]
                cap = []
                for w in raw_cap:
                    if w not in ['\limits']:
                        cap.append(w)

                gtd_stack = []
                idx = 0
                outidx = 1
                error_flag = False
                while idx < len(cap):
                    # First ch
                    if idx == 0:
                        if cap[0] in ['{', '}']:
                            print('error: {} should NOT appears at START')
                            print(line.strip())
                            sys.exit()

                        string = cap[0] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                        f_out.write(string + '\n')
                        idx += 1
                        outidx += 1

                    # Second ch
                    else:
                        # {
                        if cap[idx] == '{':

                            # { {
                            if cap[idx - 1] == '{':
                                print('error: double { appears => ', end='')
                                print(line.strip())
                                sys.exit()

                            # } {
                            if cap[idx - 1] == '}':
                                if gtd_stack:
                                    if gtd_stack[-1][0] not in ABOVE_SYMBOLS:
                                        print('error: } { not follows frac & overset ...', key, " ".join(cap))
                                        f_out.close()
                                        os.system('rm ' + gtd_path + key + '.gtd')
                                        error_flag = True
                                        break
                                    else:
                                        gtd_stack[-1][2] = 'Below'
                                        idx += 1
                                else:
                                    print('error: } { has not gtd ...', key, " ".join(cap))
                                    break

                            # @ {
                            else:
                                # [\frac, \overset] {
                                if cap[idx - 1] in ABOVE_SYMBOLS:
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Above'])
                                    idx += 1

                                # [\sqrt, \vec ...] {
                                elif cap[idx - 1] in INSIDE_SYMBOLS:
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside'])
                                    idx += 1

                                # [\underset ...] {
                                elif cap[idx - 1] in BELOW_SYMBOLS:
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Below'])
                                    idx += 1

                                # _ {
                                elif cap[idx - 1] == '_':

                                    # ['_', '^', '\\frac', '\\sqrt'] _ {
                                    if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                        print('error: ^ _ follows wrong math symbols => ', end='')
                                        print(line.strip())
                                        sys.exit()

                                    # ['\\sum', '\\int', '\\lim'] _ {
                                    elif cap[idx - 2] in BELOW_ABOVE_SYMBOLS:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Below'])
                                        idx += 1

                                    # } _ {
                                    elif cap[idx - 2] == '}':
                                        if gtd_stack[-1][0] in BELOW_ABOVE_SYMBOLS:
                                            gtd_stack[-1][2] = 'Below'
                                        else:
                                            gtd_stack[-1][2] = 'Sub'
                                        idx += 1

                                    # @ _ {
                                    else:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub'])
                                        idx += 1

                                # ^ {
                                elif cap[idx - 1] == '^':

                                    # ['_', '^', '\\frac', '\\sqrt'] ^ {
                                    if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                        print('error: ^ _ follows wrong math symbols => ', end='')
                                        print(line.strip())
                                        sys.exit()

                                    # ['\\sum', '\\int', '\\lim'] ^ {
                                    elif cap[idx - 2] in BELOW_ABOVE_SYMBOLS:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Above'])
                                        idx += 1

                                    # } ^ {
                                    elif cap[idx - 2] == '}':
                                        if gtd_stack[-1][0] in BELOW_ABOVE_SYMBOLS:
                                            gtd_stack[-1][2] = 'Above'
                                        else:
                                            gtd_stack[-1][2] = 'Sup'
                                        idx += 1

                                    # @ ^ {
                                    else:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup'])
                                        idx += 1

                                # @ { 의 모든 조건 검사후
                                else:
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside'])
                                    idx += 1

                        # }
                        elif cap[idx] == '}':

                            # } }
                            if cap[idx - 1] == '}':
                                if gtd_stack:
                                    del (gtd_stack[-1])
                                    idx += 1
                                else:
                                    idx += 1

                            # @ }
                            else:
                                idx += 1

                        # ['_', '^']
                        elif cap[idx] in SCRIPT_SYMBOLS:
                            if idx == len(cap) - 1:
                                print('error: ^ _ appers at end ...', key)
                                f_out.close()
                                os.system('rm ' + gtd_path + key + '.gtd')
                                error_flag = True
                                break
                            if cap[idx + 1] != '{':
                                print('error: ^ _ not follows { ...', key)
                                f_out.close()
                                os.system('rm ' + gtd_path + key + '.gtd')
                                error_flag = True
                                break
                            else:
                                idx += 1

                        elif cap[idx] in ['\limits']:
                            print('error: \limits happens')
                            print(line.strip())
                            sys.exit()

                        # @
                        else:
                            # { @
                            try:
                                if cap[idx - 1] == '{' and gtd_stack:
                                    string = cap[idx] + '\t' + str(outidx) + '\t' + \
                                                    gtd_stack[-1][0] + '\t' + gtd_stack[-1][1] + '\t' + gtd_stack[-1][2]
                                    f_out.write(string + '\n')
                                    outidx += 1
                                    idx += 1

                                # } @
                                elif cap[idx - 1] == '}' and gtd_stack:
                                    string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                                        1] + '\tRight'
                                    f_out.write(string + '\n')
                                    outidx += 1
                                    idx += 1
                                    del (gtd_stack[-1])

                                # @ @
                                else:
                                    parts = string.split('\t')
                                    string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tRight'
                                    f_out.write(string + '\n')
                                    outidx += 1
                                    idx += 1

                            except Exception as e:
                                print("\t\t# Error occured !!! : {} ({:d}/{:d}) : {} : {}".format(core_name, (process_num + 1), len(lines), e, line.strip()))
                                error_flag = True
                                break

                if not error_flag:
                    parts = string.split('\t')
                    string = '</s>\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tEnd'
                    f_out.write(string + '\n')
                    f_out.close()

                if (process_num + 1) // 1000 == (process_num + 1) * 1.0 / 1000:
                    print('process files', process_num)
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="Dataset type")
    parser.add_argument("--tgt_mode", required=True, choices=['TRAIN', 'TEST'], help="Target mode")
    parser.add_argument("--latex_root_path", default="../data/CROHME/", help="Root path of latex files")
    parser.add_argument("--gtd_root_path", default="../data/CROHME/", help="Root path of gtd files")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'MATHFLAT' # CROHME / 20K / MATHFLAT
TGT_MODE = 'TRAIN' # TRAIN / TEST


if __name__ == '__main__':
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--tgt_mode", TGT_MODE])
            # sys.argv.extend(["--latex_root_path", "../data/CROHME/"])
            # sys.argv.extend(["--gtd_root_path", "../data/CROHME/"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))