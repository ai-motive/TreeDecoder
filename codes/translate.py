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


def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30, rpos_beam=3):
    
    sample = []
    sample_score = []
    rpos_sample = []
    # rpos_sample_score = []
    relation_sample = []

    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k

    # current living paths and corresponding scores(-log)
    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_rpos_samples = [[]] * live_k
    hyp_relation_samples = [[]] * live_k
    # get init state, (1,n) and encoder output, (1,D,H,W)
    next_state, ctx0 = model.f_init(x)
    next_h1t = next_state
    # -1 -> My_embedding -> 0 tensor(1,m)
    next_lw = -1 * torch.ones(1, dtype=torch.int64).cuda()
    next_calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()  # (live_k,H,W)
    next_palpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    nextemb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()
    nextePmb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()    

    for ii in range(maxlen):
        ctxP = ctx0.repeat(live_k, 1, 1, 1)  # (live_k,D,H,W)
        next_lpos = ii * torch.ones(live_k, dtype=torch.int64).cuda()
        next_h01, next_ma, next_ctP, next_pa, next_palpha_past, nextemb_memory, nextePmb_memory = \
                    model.f_next_parent(params, next_lw, next_lpos, ctxP, next_state, next_h1t, next_palpha_past, nextemb_memory, nextePmb_memory, ii)
        next_ma = next_ma.cpu().numpy()
        # next_ctP = next_ctP.cpu().numpy()
        next_palpha_past = next_palpha_past.cpu().numpy()
        nextemb_memory = nextemb_memory.cpu().numpy()
        nextePmb_memory = nextePmb_memory.cpu().numpy()

        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2)) # batch * Matt * dim
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        
        next_rpos = next_ma.argsort(axis=1)[:,-rpos_beam:] # topK parent index; batch * topK
        n_gaps = nextemb_memory.shape[1]
        n_batch = nextemb_memory.shape[0]
        next_rpos_gap = next_rpos + n_gaps * np.arange(n_batch)[:, None]
        next_remb_memory = nextemb_memory.reshape([n_batch*n_gaps, nextemb_memory.shape[-1]])
        next_remb = next_remb_memory[next_rpos_gap.flatten()] # [batch*rpos_beam, emb_dim]
        rpos_scores = next_ma.flatten()[next_rpos_gap.flatten()] # [batch*rpos_beam,]

        # next_ctPC = next_ctP.repeat(1, 1, rpos_beam)
        # next_ctPC = torch.reshape(next_ctPC, (-1, next_ctP.shape[1]))
        ctxC = ctx0.repeat(live_k*rpos_beam, 1, 1, 1)
        next_ctPC = torch.zeros(next_ctP.shape[0]*rpos_beam, next_ctP.shape[1]).cuda()
        next_h01C = torch.zeros(next_h01.shape[0]*rpos_beam, next_h01.shape[1]).cuda()
        next_calpha_pastC = torch.zeros(next_calpha_past.shape[0]*rpos_beam, next_calpha_past.shape[1], next_calpha_past.shape[2]).cuda()
        for bidx in range(next_calpha_past.shape[0]):
            for ridx in range(rpos_beam):
                next_ctPC[bidx*rpos_beam+ridx] = next_ctP[bidx]
                next_h01C[bidx*rpos_beam+ridx] = next_h01[bidx]
                next_calpha_pastC[bidx*rpos_beam+ridx] = next_calpha_past[bidx]
        next_remb = torch.from_numpy(next_remb).cuda()

        next_lp, next_rep, next_state, next_h1t, next_ca, next_calpha_past, next_re = \
                    model.f_next_child(params, next_remb, next_ctPC, ctxC, next_h01C, next_calpha_pastC)

        next_lp = next_lp.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_h1t = next_h1t.cpu().numpy()
        next_calpha_past = next_calpha_past.cpu().numpy()
        next_re = next_re.cpu().numpy()

        hyp_scores = np.tile(hyp_scores[:, None], [1, rpos_beam]).flatten()
        cand_scores = hyp_scores[:, None] - np.log(next_lp+1e-10)- np.log(rpos_scores+1e-10)[:,None]
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        voc_size = next_lp.shape[1]
        trans_indices = ranks_flat // voc_size
        trans_indicesP = ranks_flat // (voc_size*rpos_beam)
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        # update paths
        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_rpos_samples = []
        new_hyp_relation_samples = []
        new_hyp_states = []
        new_hyp_h1ts = []
        new_hyp_calpha_past = []
        new_hyp_palpha_past = []
        new_hyp_emb_memory = []
        new_hyp_ePmb_memory = []
        
        for idx, [ti, wi, tPi] in enumerate(zip(trans_indices, word_indices, trans_indicesP)):
            new_hyp_samples.append(hyp_samples[tPi]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_rpos_samples.append(hyp_rpos_samples[tPi]+[next_rpos.flatten()[ti]])
            new_hyp_relation_samples.append(hyp_relation_samples[tPi]+[next_re[ti]])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_h1ts.append(copy.copy(next_h1t[ti]))
            new_hyp_calpha_past.append(copy.copy(next_calpha_past[ti]))
            new_hyp_palpha_past.append(copy.copy(next_palpha_past[tPi]))
            new_hyp_emb_memory.append(copy.copy(nextemb_memory[tPi]))
            new_hyp_ePmb_memory.append(copy.copy(nextePmb_memory[tPi]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_rpos_samples = []
        hyp_relation_samples = []
        hyp_states = []
        hyp_h1ts = []
        hyp_calpha_past = []
        hyp_palpha_past = []
        hyp_emb_memory = []
        hyp_ePmb_memory = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0: # <eol>
                sample_score.append(new_hyp_scores[idx])
                sample.append(new_hyp_samples[idx])
                rpos_sample.append(new_hyp_rpos_samples[idx])
                relation_sample.append(new_hyp_relation_samples[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_scores.append(new_hyp_scores[idx])
                hyp_samples.append(new_hyp_samples[idx])
                hyp_rpos_samples.append(new_hyp_rpos_samples[idx])
                hyp_relation_samples.append(new_hyp_relation_samples[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_h1ts.append(new_hyp_h1ts[idx])
                hyp_calpha_past.append(new_hyp_calpha_past[idx])
                hyp_palpha_past.append(new_hyp_palpha_past[idx])
                hyp_emb_memory.append(new_hyp_emb_memory[idx])
                hyp_ePmb_memory.append(new_hyp_ePmb_memory[idx])   
                    
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_lw = np.array([w[-1] for w in hyp_samples])  # each path's final symbol, (live_k,)
        next_state = np.array(hyp_states)  # h2t, (live_k,n)
        next_h1t = np.array(hyp_h1ts)
        next_calpha_past = np.array(hyp_calpha_past)  # (live_k,H,W)
        next_palpha_past = np.array(hyp_palpha_past)
        nextemb_memory = np.array(hyp_emb_memory)
        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2))
        nextePmb_memory = np.array(hyp_ePmb_memory)
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        next_lw = torch.from_numpy(next_lw).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_h1t = torch.from_numpy(next_h1t).cuda()
        next_calpha_past = torch.from_numpy(next_calpha_past).cuda()
        next_palpha_past = torch.from_numpy(next_palpha_past).cuda()
        nextemb_memory = torch.from_numpy(nextemb_memory).cuda()
        nextePmb_memory = torch.from_numpy(nextePmb_memory).cuda()

    return sample_score, sample, rpos_sample, relation_sample


def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])

    print('total words/phones', len(lexicon))
    return lexicon


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