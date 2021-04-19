import argparse
import sys
import os


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def gen_voc(infile, vocfile):
    vocab=set()
    with open(infile) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print('illegal line: ', line)
                continue
            (title,label) = parts
            for w in label.split():
                if w not in vocab:
                    vocab.add(w)
    with open(vocfile,'w') as fout:
        for i, w in enumerate(vocab):
            fout.write('{}\t{}\n'.format(w,i+1))
        # fout.write('<eol>\t0\n')
        fout.write('<s>\t{}\n'.format(len(vocab)+1))
        fout.write('</s>\t0\n')

    return True

def main(args):
    cation_path = os.path.join(_this_folder_, args.dataset_type, 'caption', 'total_caption.txt')
    dict_path = os.path.join(_this_folder_, args.dataset_type, 'dictionary.txt')
    gen_ = gen_voc(cation_path, dict_path)
    if gen_:
        print("Generated dictonary : {}".format(dict_path))
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