import argparse
import copy
import numpy as np
import os
import time
import sys
import torch
from codes.data_iterator import dataIterator, dataIterator_test
from codes.encoder_decoder import Encoder_Decoder
from codes.utils import load_dict, gen_sample, compute_wer, compute_sacc, parse_to_latexes
from datetime import datetime


# Note:
#   here model means Encoder_Decoder -->  WAP_model
#   x means a sample not a batch(or batch_size = 1),and x's shape should be (1,1,H,W),type must be Variable
#   live_k is just equal to k -dead_k(except the begin of sentence:live_k = 1,dead_k = 0,so use k-dead_k to represent the number of alive paths in beam search)


def main_test(args):
    concat_dataset_path, test_path, model_path, dictionary_target, dictionary_retarget, fea, output_path, k = \
        args.concat_dataset_path, args.test_path, args.model_path, args.dictionary_target, args.dictionary_retarget, args.fea, args.output_path, args.k

    # Paths for train, test
    if args.dataset_type == 'CROHME':
        concat_dataset_path = '../data/CROHME/'
        img_path, cptn_path = os.path.join(concat_dataset_path, 'image/'), os.path.join(concat_dataset_path, 'caption/')
        test_img_pkl_path = os.path.join(img_path, 'offline-test.pkl')
        test_label_pkl_path = os.path.join(cptn_path, 'test_caption_label_gtd.pkl')
        test_align_pkl_path = os.path.join(cptn_path, 'test_caption_label_align_gtd.pkl')
    elif args.dataset_type == 'MATHFLAT':
        test_img_pkl_path = os.path.join(args.test_path, 'offline-test.pkl')
        test_label_pkl_path = os.path.join(args.test_path, 'test_caption_label.pkl')
        test_align_pkl_path = os.path.join(args.test_path, 'test_caption_align.pkl')

    valid_datasets = [test_img_pkl_path, test_label_pkl_path, test_align_pkl_path]

    # set parameters
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = args.K  ## num class : 106
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 1
    params['Kre'] = args.Kre  ## num relation
    params['mre'] = 256

    maxlen = args.maxlen
    params['maxlen'] = maxlen

    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # enable CUDA
    model.cuda()

    # load source dictionary and invert
    worddicts = load_dict(dictionary_target)
    print('total chars', len(worddicts))
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    reworddicts = load_dict(dictionary_retarget)
    print('total relations', len(reworddicts))
    reworddicts_r = [None] * len(reworddicts)
    for kk, vv in reworddicts.items():
        reworddicts_r[vv] = kk

    valid, valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2], worddicts,
                                         reworddicts,
                                         batch_size=args.batch_size, batch_Imagesize=800000,
                                         maxlen=maxlen, maxImagesize=500000)

    # change model's mode to eval
    model.eval()
    model_date = datetime.today().strftime("%y%m%d")

    valid_out_path = os.path.join(output_path, model_date, 'symbol_relation/')
    valid_malpha_path = os.path.join(output_path, model_date, 'memory_alpha/')
    if not os.path.exists(valid_out_path):
        os.makedirs(valid_out_path)
    if not os.path.exists(valid_malpha_path):
        os.makedirs(valid_malpha_path)

    print('Decoding ... ')
    ud_epoch = time.time()
    model.eval()
    rec_mat = {}
    label_mat = {}
    rec_re_mat = {}
    label_re_mat = {}
    rec_ridx_mat = {}
    label_ridx_mat = {}
    with torch.no_grad():
        valid_count_idx = 0
        for x, ly, ry, re, ma, lp, rp in valid:
            for xx, lyy, ree, rpp in zip(x, ly, re, rp):
                xx_pad = xx.astype(np.float32) / 255.
                xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                score, sample, malpha_list, relation_sample = \
                    gen_sample(model, xx_pad, params, False, k=k, maxlen=maxlen, rpos_beam=3)

                key = valid_uid_list[valid_count_idx]
                rec_mat[key] = []
                label_mat[key] = lyy
                rec_re_mat[key] = []
                label_re_mat[key] = ree
                rec_ridx_mat[key] = []
                label_ridx_mat[key] = rpp
                if len(score) == 0:
                    rec_mat[key].append(0)
                    rec_re_mat[key].append(0)  # End
                    rec_ridx_mat[key].append(0)
                else:
                    score = score / np.array([len(s) for s in sample])
                    min_score_index = score.argmin()
                    ss = sample[min_score_index]
                    rs = relation_sample[min_score_index]
                    mali = malpha_list[min_score_index]
                    fpp_sample = open(valid_out_path + valid_uid_list[valid_count_idx] + '.txt', 'w')  ##
                    file_malpha_sample = valid_malpha_path + valid_uid_list[valid_count_idx] + '_malpha.txt'  ##
                    for i, [vv, rv] in enumerate(zip(ss, rs)):
                        if vv == 0:
                            rec_mat[key].append(vv)
                            rec_re_mat[key].append(0)  # End
                            string = worddicts_r[vv] + '\tEnd\n'  ##
                            fpp_sample.write(string)  ##
                            break
                        else:
                            if i == 0:
                                rec_mat[key].append(vv)
                                rec_re_mat[key].append(6)  # Start
                                string = worddicts_r[vv] + '\tStart\n'  ##
                            else:
                                rec_mat[key].append(vv)
                                rec_re_mat[key].append(rv)
                                string = worddicts_r[vv] + '\t' + reworddicts_r[rv] + '\n'  ##
                            fpp_sample.write(string)  ##

                    ma_idx_list = np.array(mali).astype(np.int64)
                    ma_idx_list[-1] = int(len(ma_idx_list) - 1)
                    rec_ridx_mat[key] = ma_idx_list
                    np.savetxt(file_malpha_sample, np.array(mali))  ##
                    fpp_sample.close()  ##

                valid_count_idx = valid_count_idx + 1

            print('{}/{}-th test data processed !!!'.format(valid_count_idx, len(valid_uid_list)))

    print('test set decode done')
    ud_epoch = (time.time() - ud_epoch) / 60.
    print('epoch cost time ... ', ud_epoch)

    # Evalute perf.
    valid_cer_out = compute_wer(rec_mat, label_mat)
    valid_cer = 100. * valid_cer_out[0]
    valid_recer_out = compute_wer(rec_re_mat, label_re_mat)
    valid_recer = 100. * valid_recer_out[0]
    valid_ridxcer_out = compute_wer(rec_ridx_mat, label_ridx_mat)
    valid_ridxcer = 100. * valid_ridxcer_out[0]
    valid_exprate = compute_sacc(rec_mat, label_mat, rec_ridx_mat, label_ridx_mat, rec_re_mat, label_re_mat,
                                 worddicts_r, reworddicts_r)
    valid_exprate = 100. * valid_exprate
    print('Valid CER: %.2f%%, relation_CER: %.2f%%, rpos_CER: %.2f%%, ExpRate: %.2f%%'
          % (valid_cer, valid_recer, valid_ridxcer, valid_exprate))

    return True

def main_inference(args):
    concat_dataset_path, test_path, model_path, dictionary_target, dictionary_retarget, fea, output_path, k = \
        args.concat_dataset_path, args.test_path, args.model_path, args.dictionary_target, args.dictionary_retarget, args.fea, args.output_path, args.k

    # Paths for test
    if args.dataset_type == 'CROHME':
        concat_dataset_path = '../data/CROHME/'
        img_path, cptn_path = os.path.join(concat_dataset_path, 'image/'), os.path.join(concat_dataset_path, 'caption/')
        test_img_pkl_path = os.path.join(img_path, 'offline-test.pkl')

    elif args.dataset_type == 'MATHFLAT':
        test_img_pkl_path = os.path.join(args.test_path, 'offline-test.pkl')

    valid_datasets = [test_img_pkl_path]

    # set parameters
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = args.K  ## num class : 106
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 1
    params['Kre'] = args.Kre  ## num relation
    params['mre'] = 256

    maxlen = args.maxlen
    params['maxlen'] = maxlen

    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # enable CUDA
    model.cuda()

    # load source dictionary and invert
    worddicts = load_dict(dictionary_target)
    print('total chars', len(worddicts))
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    reworddicts = load_dict(dictionary_retarget)
    print('total relations', len(reworddicts))
    reworddicts_r = [None] * len(reworddicts)
    for kk, vv in reworddicts.items():
        reworddicts_r[vv] = kk

    valid, valid_uid_list = dataIterator_test(valid_datasets[0],
                                              batch_size=args.batch_size, batch_Imagesize=800000,
                                              maxImagesize=500000)

    # change model's mode to eval
    model.eval()

    print('Decoding ... ')
    ud_epoch = time.time()
    model.eval()
    rec_mat = {}
    rec_re_mat = {}
    rec_ridx_mat = {}
    with torch.no_grad():
        valid_count_idx = 0
        for x in valid:
            for xx in x:
                xx_pad = xx.astype(np.float32) / 255.
                xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                score, sample, malpha_list, relation_sample = \
                    gen_sample(model, xx_pad, params, False, k=k, maxlen=maxlen, rpos_beam=3)

                key = valid_uid_list[valid_count_idx]
                rec_mat[key] = []
                rec_re_mat[key] = []
                rec_ridx_mat[key] = []
                if len(score) == 0:
                    rec_mat[key].append(0)
                    rec_re_mat[key].append(0)  # End
                    rec_ridx_mat[key].append(0)
                else:
                    score = score / np.array([len(s) for s in sample])
                    min_score_index = score.argmin()
                    ss = sample[min_score_index]
                    rs = relation_sample[min_score_index]
                    mali = malpha_list[min_score_index]
                    for i, [vv, rv] in enumerate(zip(ss, rs)):
                        if vv == 0:
                            rec_mat[key].append(vv)
                            rec_re_mat[key].append(0)  # End
                            break
                        else:
                            if i == 0:
                                rec_mat[key].append(vv)
                                rec_re_mat[key].append(6)  # Start
                            else:
                                rec_mat[key].append(vv)
                                rec_re_mat[key].append(rv)

                    ma_idx_list = np.array(mali).astype(np.int64)
                    ma_idx_list[-1] = int(len(ma_idx_list) - 1)
                    rec_ridx_mat[key] = ma_idx_list

                valid_count_idx = valid_count_idx + 1

            print('{}/{}-th test data processed !!!'.format(valid_count_idx, len(valid_uid_list)))

    print('test set decode done')
    ud_epoch = (time.time() - ud_epoch) / 60.
    print('epoch cost time ... ', ud_epoch)

    # Parse to latex
    latexes = parse_to_latexes(rec_mat, rec_ridx_mat, rec_re_mat, worddicts_r, reworddicts_r)
    print(latexes)

    return True

def main(args):
    if args.op_mode == 'TEST':
        main_test(args)
    elif args.op_mode == 'INFERENCE':
        main_inference(args)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_mode", required=True, choices=['TEST', 'INFERENCE'], help="operation mode")
    parser.add_argument("--dataset_type", required=True, choices=['CROHME', '20K', 'MATHFLAT'], help="dataset type")
    parser.add_argument("--concat_dataset_path", type=str, help="Concated dataset path")
    parser.add_argument("--test_path", type=str, help="test data folder path")
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--maxlen', type=int, default=200, help='maximum-label-length')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument("--model_path", default="../train/models/210418/WAP_params_last.pkl", type=str, help="pretrain model path")
    parser.add_argument("--dictionary_target", default="../data/CROHME/dictionary.txt", type=str, help="dictionary of target class")
    parser.add_argument("--dictionary_retarget", default="../data/CROHME/relation_dictionary.txt", type=str, help="dictionary of relation target class")
    parser.add_argument("--fea", default="../data/CROHME/image/offline-test.pkl", type=str, help="image feature file")
    parser.add_argument("--output_path", default="../test/", type=str, help="test result path")

    """ Model Architecture """
    parser.add_argument('--K', type=int, default=106, help='number of character label')  # 112
    parser.add_argument('--Kre', type=int, default=8, help='number of character relation')

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'INFERENCE' # TEST / INFERENCE
DATASET_TYPE = 'MATHFLAT' # CROHME / 20K / MATHFLAT


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--concat_dataset_path", '/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/total/concat/tree_math_gt/'])
            sys.argv.extend(["--test_path", '/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/total/test/tree_math_gt/'])
            # sys.argv.extend(["--test_path", '/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/total/train/tree_math_gt/']) # for verif.
            sys.argv.extend(["--model_path", '../train/models/210510/WAP_params.pkl'])
            sys.argv.extend(["--dictionary_target", '/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/total/concat/tree_math_gt/dictionary.txt'])
            sys.argv.extend(["--dictionary_retarget", '/HDD/Datasets/mathflat_problems/Output_supervisely_V4.1/total/concat/tree_math_gt/re_dictionary.txt'])
            sys.argv.extend(["--output_path", '../test/'])
            sys.argv.extend(["--batch_size", '6'])
            sys.argv.extend(["--K", '156'])
            sys.argv.extend(["--k", '3'])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))