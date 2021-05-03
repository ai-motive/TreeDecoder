import os
import sys
import json
import argparse
import subprocess
# from sklearn.model_selection import train_test_split
from utility import general_utils as utils
from utility.str_utils import replace_string_from_dict
from utility import multi_process
from codes import latex2gtd


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main_generate_split_cptn(ini, common_info, logger=None):
    """
    craft_train_path, craft_test_path 경로의 파일과 ann/ 파일간의
    일치하는 데이터를  추출하여 cptn_path에 저장한다.
    """

    # Init. path variables
    for key, val in ini.items():
        globals()[key] = replace_string_from_dict(val, common_info)

    utils.folder_exists(total_cptn_path, create_=True)
    utils.folder_exists(train_cptn_path, create_=True)
    utils.folder_exists(test_cptn_path, create_=True)

    ann_fnames = sorted(utils.get_filenames(ann_path, extensions=utils.META_EXTENSION))
    logger.info(" [GENERATE_SPLIT_CPTN] # Total file number to be processed: {:d}.".format(len(ann_fnames)))

    tree_gt_list = []
    for idx, ann_fname in enumerate(ann_fnames):
        logger.info(" [GENERATE_SPLIT_CPTN] # Processing {} ({:d}/{:d})".format(ann_fname, (idx+1), len(ann_fnames)))

        # Load json
        _, ann_core_name, _ = utils.split_fname(ann_fname)
        ann_core_name = ann_core_name.replace('.jpg', '')
        with open(ann_fname) as json_file:
            json_data = json.load(json_file)
            objects = json_data['objects']
            # pprint.pprint(objects)

        texts = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != common_info['tgt_class']:
                continue

            text = obj['description']
            texts.append(text)

        for t_idx, text in enumerate(texts):
            tree_gt_list.append("".join([ann_core_name + '_crop_' + '{0:03d}'.format(t_idx), '\t', text + '\n']))

    with open(os.path.join(total_cptn_path, "total_caption.txt"), "w", encoding="utf8") as f:
        for i in range(len(tree_gt_list)):
            gt = tree_gt_list[i]
            f.write("{}".format(gt))

    # Match CRAFT TRAIN & TEST
    craft_train_list = sorted(utils.get_filenames(craft_train_path, extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(craft_test_path, extensions=utils.TEXT_EXTENSIONS))

    tree_train_list = []
    tree_test_list = []
    for tree_gt in tree_gt_list:
        gt_fname = tree_gt.split('\t')[0]
        gt_core_name = gt_fname.split('_crop')[0]
        tree_fname = gt_core_name + '.txt'
        match_train_fname = os.path.join(craft_train_path, 'gt_' + tree_fname)
        match_test_fname = os.path.join(craft_test_path, 'gt_' + tree_fname)

        if match_train_fname in craft_train_list:
            tree_train_list.append(tree_gt)
        elif match_test_fname in craft_test_list:
            tree_test_list.append(tree_gt)

    # Save train.txt file
    train_fpath = os.path.join(train_cptn_path, 'train_caption.txt')
    with open(train_fpath, 'w') as f:
        f.write(''.join(tree_train_list))

    test_fpath = os.path.join(test_cptn_path, 'test_caption.txt')
    with open(test_fpath, 'w') as f:
        f.write(''.join(tree_test_list))

    logger.info(" [GENERATE_SPLIT_CPTN] # Train : Test ratio -> {} % : {} %".format(int(len(tree_train_list) / len(tree_gt_list) * 100),
                                                                                    int(len(tree_test_list) / len(tree_gt_list) * 100)))
    logger.info(" [GENERATE_SPLIT_CPTN] # Train : Test size  -> {} : {}".format(len(tree_train_list), len(tree_test_list)))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_generate_gtd(ini, common_info, logger=None):
    """
        train_cptn_path, test_cptn_path 파일을 gtd로 변환하여
        train_gtd_path, test_gtd_path에 저장한다.
        """

    # Init. path variables
    for key, val in ini.items():
        globals()[key] = replace_string_from_dict(val, common_info)

    utils.folder_exists(train_gtd_path, create_=True)
    utils.folder_exists(test_gtd_path, create_=True)

    for tgt_mode in ['TRAIN', 'TEST']:
        if tgt_mode == 'TRAIN':
            latex_path = train_cptn_path
            gtd_path = train_gtd_path
        elif tgt_mode == 'TEST':
            latex_path = test_cptn_path
            gtd_path =  test_gtd_path

        generate_gtd_args = [
            '--dataset_type', common_info['dataset_type'],
            '--tgt_mode', tgt_mode,
            '--latex_root_path', latex_path,
            '--gtd_root_path', gtd_path,

            # '--dataset_type', 'CROHME',
            # '--tgt_mode', tgt_mode,
            # '--latex_root_path', "../data/CROHME/latex/",
            # '--gtd_root_path', "../data/CROHME/latex/",
        ]
        latex2gtd.main(latex2gtd.parse_arguments(generate_gtd_args))

    return True

def main_crop(ini, model_dir=None, logger=None):
    craft_train_list = sorted(utils.get_filenames(ini['craft_train_path'], extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(ini['craft_test_path'], extensions=utils.TEXT_EXTENSIONS))
    logger.info(" [CRAFT-TRAIN GT] # Total gt number to be processed: {:d}.".format(len(craft_train_list)))

    for craft_list in [craft_train_list, craft_test_list]:
        if craft_list is craft_train_list:
            tar_mode = 'TRAIN'
        elif craft_list is craft_test_list:
            tar_mode = 'TEST'

        available_cpus = len(os.sched_getaffinity(0))
        mp_inputs = [(craft_fpath, ini, tar_mode) for file_idx, craft_fpath in enumerate(craft_list)]

        # Multiprocess func.
        multi_process.run(func=load_craft_gt_and_save_crop_images, data=mp_inputs,
                          n_workers=available_cpus, n_tasks=len(craft_list), max_queue_size=len(craft_list))

    return True

def load_craft_gt_and_save_crop_images(craft_fpath, ini, tar_mode, print_=False):
    # load craft gt. file
    with open(craft_fpath, "r", encoding="utf8") as f:
        craft_infos = f.readlines()
        for tl_idx, craft_info in enumerate(craft_infos):
            box = craft_info.split(',')[:8]
            box = [int(pos) for pos in box]
            x1, y1, x3, y3 = box[0], box[1], box[4], box[5]

            _, core_name, _ = utils.split_fname(craft_fpath)
            img_fname = core_name.replace('gt_', '')

            if tar_mode == 'TRAIN':
                raw_img_path = os.path.join(ini['train_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(ini['train_crop_path'],
                                         img_fname + '_crop_' + '{0:03d}'.format(tl_idx) + '.jpg')
            elif tar_mode == 'TEST':
                raw_img_path = os.path.join(ini['test_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(ini['test_crop_path'],
                                         img_fname + '_crop_' + '{0:03d}'.format(tl_idx) + '.jpg')

            if not (utils.file_exists(raw_img_path, print_=True)):
                print("  # Raw image doesn't exists at {}".format(raw_img_path))
                continue

            img = utils.imread(raw_img_path, color_fmt='RGB')
            crop_img = img[y1:y3, x1:x3]

            if utils.file_exists(rst_fpath):
                print("  # Save image already exists at {}".format(rst_fpath))
                pass
            else:
                utils.imwrite(crop_img, rst_fpath)
                print("  #  ({:d}/{:d}) Saved at {} ".format(tl_idx, len(craft_infos), rst_fpath))

    return True

def main_create(ini, model_dir=None, logger=None):
    for tar_mode in ['TRAIN', 'TEST']:
        if tar_mode == 'TRAIN':
            crop_img_path = os.path.join(ini['train_gt_path'], 'crop_img')
            gt_fpath = os.path.join(ini['train_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(ini['train_lmdb_path'])
        elif tar_mode == 'TEST':
            crop_img_path = os.path.join(ini['test_gt_path'], 'crop_img')
            gt_fpath = os.path.join(ini['test_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(ini['test_lmdb_path'])

        logger.info(" [CREATE-{}] # Create lmdb dataset".format(tar_mode))
        create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)

    return True

def main_merge(ini, model_dir=None, logger=None):
    global src_train_gt_path, src_test_gt_path, dst_train_gt_path, dst_test_gt_path
    utils.folder_exists(ini['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(ini['dataset_path']) if dataset != 'total']
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    # Process total files
    train_gt_text_paths = []
    test_gt_text_paths = []
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            src_train_path, src_test_path = os.path.join(ini['dataset_path'], dir_name, 'train'), os.path.join(ini['dataset_path'], dir_name, 'test')
            src_train_crop_img_path = os.path.join(src_train_path, 'crnn_gt/crop_img/')
            src_test_crop_img_path = os.path.join(src_test_path, 'crnn_gt/crop_img/')

            dst_train_path, dst_test_path = os.path.join(ini['total_dataset_path'], 'train'), os.path.join(ini['total_dataset_path'], 'test')
            dst_train_crop_img_path = os.path.join(dst_train_path, 'crnn_gt/crop_img/')
            dst_test_crop_img_path = os.path.join(dst_test_path, 'crnn_gt/crop_img/')

            if utils.folder_exists(dst_train_crop_img_path) and utils.folder_exists(dst_test_crop_img_path):
                logger.info(" # Already {} is exist".format(ini['total_dataset_path']))
            else:
                utils.folder_exists(dst_train_crop_img_path, create_=True), utils.folder_exists(dst_test_crop_img_path, create_=True)

            # Apply symbolic link for gt & img path
            for tar_mode in ['TRAIN', 'TEST']:
                if tar_mode is 'TRAIN':
                    src_crop_img_path = src_train_crop_img_path
                    dst_crop_img_path = dst_train_crop_img_path
                elif tar_mode is 'TEST':
                    src_crop_img_path = src_test_crop_img_path
                    dst_crop_img_path = dst_test_crop_img_path

                # link img_path
                src_crop_imgs = sorted(utils.get_filenames(src_crop_img_path, extensions=utils.IMG_EXTENSIONS))
                dst_crop_imgs = sorted(utils.get_filenames(dst_crop_img_path, extensions=utils.IMG_EXTENSIONS))

                src_crop_fnames = [utils.split_fname(crop_img)[1] for crop_img in src_crop_imgs]
                dst_crop_fnames = [utils.split_fname(crop_img)[1] for crop_img in dst_crop_imgs]
                if any(src_fname not in dst_crop_fnames for src_fname in src_crop_fnames):
                    img_sym_cmd = 'find {} -name "*.jpg" -exec ln {} {} \;'.format(src_crop_img_path, '{}', dst_crop_img_path) # link each files
                    # img_sym_cmd = 'ln "{}"* "{}"'.format(src_crop_img_path, dst_crop_img_path)  # argument is long
                    subprocess.call(img_sym_cmd, shell=True)
                    logger.info(" # Link img files {} -> {}.".format(src_crop_img_path, dst_crop_img_path))
                else:
                    logger.info(" # Link img files already generated : {}.".format(dst_crop_img_path))

            # Add to list all label files
            for tar_mode in ['TRAIN', 'TEST']:
                if tar_mode == 'TRAIN':
                    src_train_gt_path = os.path.join(src_train_path, 'crnn_gt', 'labels.txt')
                    train_gt_text_paths.append(src_train_gt_path)

                    dst_train_gt_path = os.path.join(dst_train_path, 'crnn_gt', 'labels.txt')

                elif tar_mode == 'TEST':
                    src_test_gt_path = os.path.join(src_test_path, 'crnn_gt', 'labels.txt')
                    test_gt_text_paths.append(src_test_gt_path)

                    dst_test_gt_path = os.path.join(dst_test_path, 'crnn_gt', 'labels.txt')

        logger.info(" # Train gt paths : {}".format(train_gt_text_paths))
        logger.info(" # Test gt paths : {}".format(test_gt_text_paths))

        # Merge all label files
        with open(dst_train_gt_path, 'w') as outfile:
            for fpath in train_gt_text_paths:
                with open(fpath) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(dst_test_gt_path, 'w') as outfile:
            for fpath in test_gt_text_paths:
                with open(fpath) as infile:
                    for line in infile:
                        outfile.write(line)

        logger.info(" # Train & Test gt files are merged !!!")

        for tar_mode in ['TRAIN', 'TEST']:
            logger.info(" [CREATE-{}] # Create lmdb dataset".format(tar_mode))
            if tar_mode == 'TRAIN':
                crop_img_path = dst_train_crop_img_path
                gt_fpath = dst_train_gt_path
                lmdb_path = ini['total_train_lmdb_path']
            elif tar_mode == 'TEST':
                crop_img_path = dst_test_crop_img_path
                gt_fpath = dst_test_gt_path
                lmdb_path = ini['total_test_lmdb_path']

            create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)
        logger.info(" [CREATE-ALL] # Create all lmdb dataset")

    return True

def main_train(ini, model_dir=None, logger=None):
    cuda_ids = ini['cuda_ids'].split(',')
    train_args = ['--train_data', ini['train_lmdb_path'],
                  '--valid_data', ini['test_lmdb_path'],
                  '--cuda', ini['cuda'],
                  '--cuda_ids', cuda_ids,
                  '--workers', ini['workers'],
                  '--batch_size', ini['batch_size'],
                  '--num_iter', ini['num_iter'],
                  # '--saved_model', ini['saved_model'],
                  '--select_data', ini['select_data'],
                  '--Transformation', ini['Transformation'],
                  '--FeatureExtraction', ini['FeatureExtraction'],
                  '--SequenceModeling', ini['SequenceModeling'],
                  '--Prediction', ini['Prediction'],
                  '--data_filtering_off',
                  # '--batch_ratio', ini['batch_ratio'],
                  '--batch_max_length', ini['batch_max_length'],
                  '--imgH', ini['imgH'],
                  '--imgW', ini['imgW'],
                  '--PAD',
                  '--hidden_size', ini['hidden_size']]

    train.main(train_args)

    return True

def main_test(ini, model_dir=None, logger=None):

    return True


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE_SPLIT_CPTN':
        main_generate_split_cptn(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'GENERATE_GTD':
        main_generate_gtd(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'CROP_IMG':
        main_crop(ini[args.op_mode], logger=logger)
    elif args.op_mode == 'CREATE_LMDB':
        main_create(ini[args.op_mode], logger=logger)
    elif args.op_mode == 'MERGE':
        main_merge(ini[args.op_mode], logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini[args.op_mode], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini[args.op_mode], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
        main_test(ini['TEST'], model_dir, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['GENERATE_SPLIT_CPTN', 'GENERATE_GTD', 'CROP_IMG', 'CREATE_LMDB', 'MERGE', 'TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'GENERATE_GTD' # GENERATE_SPLIT_CPTN / GENERATE_GTD / CROP_IMG / CREATE_LMDB or MERGE / TRAIN / TEST / TRAIN_TEST
INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))


