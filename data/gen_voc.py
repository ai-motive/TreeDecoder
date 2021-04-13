import sys
import os

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


dataset_type = '20K'    # CHROHME / 20K


if __name__ == '__main__':
    if len(sys.argv) != 3:
        cation_path = os.path.join(dataset_type, 'total_caption.txt')
        dict_path = os.path.join(dataset_type, 'dictionary.txt')
        gen_voc(cation_path, dict_path)
        sys.exit(0)
    else:
        gen_voc(sys.argv[1], sys.argv[2])