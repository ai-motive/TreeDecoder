#!/usr/bin/env python
import argparse
import os
import sys
import pickle as pkl
import numpy


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

BELOW_ABOVE_SYMBOLS = ['\\sum', '\\int', '\\lim']
SCRIPT_SYMBOLS = ['_', '^']


def main(args):
    latex_root_path = '../data/{}/'.format(args.dataset_type)
    gtd_root_path = '../data/{}/'.format(args.dataset_type)

    latex_files = ['train_caption.txt', 'test_caption.txt']
    for latexF in latex_files:
        latex_file = os.path.join(latex_root_path, 'caption', latexF)
        gtd_path = os.path.join(gtd_root_path, 'caption', latexF[:-4] + '/')
        if not os.path.exists(gtd_path):
            os.mkdir(gtd_path)

        with open(latex_file) as f:
            lines = f.readlines()
            for process_num, line in enumerate(lines):
                if '\sqrt [' in line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print('error: invalid latex caption ...', line)
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
                    if idx == 0:  # 시작 문자
                        if cap[0] in ['{', '}']:
                            print('error: {} should NOT appears at START')
                            print(line.strip())
                            sys.exit()
                        string = cap[0] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                        f_out.write(string + '\n')
                        idx += 1
                        outidx += 1
                    else:
                        if cap[idx] == '{':
                            if cap[idx - 1] == '{':
                                print('error: double { appears => ', end='')
                                print(line.strip())
                                # sys.exit()
                                break  ##
                            if cap[idx - 1] == '}':
                                if gtd_stack[-1][0] != '\\frac':
                                    print('error: } { not follows frac ...', key)
                                    f_out.close()
                                    os.system('rm ' + gtd_path + key + '.gtd')
                                    error_flag = True
                                    break
                                else:
                                    gtd_stack[-1][2] = 'Below'
                                    idx += 1
                            else:
                                if cap[idx - 1] == '\\frac':
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Above'])
                                    idx += 1
                                elif cap[idx - 1] == '\\sqrt':
                                    gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside'])
                                    idx += 1
                                elif cap[idx - 1] == '_':
                                    if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                        print('error: ^ _ follows wrong math symbols => ', end='')
                                        print(line.strip())
                                        sys.exit()
                                    elif cap[idx - 2] in BELOW_ABOVE_SYMBOLS:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Below'])
                                        idx += 1
                                    elif cap[idx - 2] == '}':
                                        if gtd_stack[-1][0] in BELOW_ABOVE_SYMBOLS:
                                            gtd_stack[-1][2] = 'Below'
                                        else:
                                            gtd_stack[-1][2] = 'Sub'
                                        idx += 1
                                    else:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub'])
                                        idx += 1
                                elif cap[idx - 1] == '^':
                                    if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                        print('error: ^ _ follows wrong math symbols => ', end='')
                                        print(line.strip())
                                        sys.exit()
                                    elif cap[idx - 2] in BELOW_ABOVE_SYMBOLS:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Above'])
                                        idx += 1
                                    elif cap[idx - 2] == '}':
                                        if gtd_stack[-1][0] in BELOW_ABOVE_SYMBOLS:
                                            gtd_stack[-1][2] = 'Above'
                                        else:
                                            gtd_stack[-1][2] = 'Sup'
                                        idx += 1
                                    else:
                                        gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup'])
                                        idx += 1
                                else:
                                    print('error: { follows unknown math symbols ...', key)
                                    f_out.close()
                                    os.system('rm ' + gtd_path + key + '.gtd')
                                    error_flag = True
                                    break
                        elif cap[idx] == '}':
                            if cap[idx - 1] == '}':
                                del (gtd_stack[-1])
                            idx += 1
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
                        else:
                            if cap[idx - 1] == '{':
                                string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                                    1] + '\t' + gtd_stack[-1][2]
                                f_out.write(string + '\n')
                                outidx += 1
                                idx += 1
                            elif cap[idx - 1] == '}':
                                string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                                    1] + '\tRight'
                                f_out.write(string + '\n')
                                outidx += 1
                                idx += 1
                                del (gtd_stack[-1])
                            else:
                                parts = string.split('\t')
                                string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tRight'
                                f_out.write(string + '\n')
                                outidx += 1
                                idx += 1
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