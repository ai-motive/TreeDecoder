import os
import sys
import json
import argparse
import subprocess
# from sklearn.model_selection import train_test_split
from utility import general_utils as utils
from utility.str_utils import replace_string_from_dict
from utility import multi_process
from codes import latex2gtd, prepare_label, train_wap
from data import gen_pkl, gen_voc


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main_generate_split_cptn(ini, common_info, logger=None):
    """
    craft_train_path, craft_test_path 경로의 파일과 ann/ 파일간의
    일치하는 데이터를  추출하여 cptn_path에 저장한다.
    """

    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    utils.folder_exists(vars['total_cptn_path'], create_=True)
    utils.folder_exists(vars['train_cptn_path'], create_=True)
    utils.folder_exists(vars['test_cptn_path'], create_=True)

    ann_fnames = sorted(utils.get_filenames(vars['ann_path'], extensions=utils.META_EXTENSION))
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

            strip_latex = text.replace(' ', '')
            if strip_latex[:2] == '{}':
                strip_latex = strip_latex[2:]

            latex_parts = latex2gtd.split_string_to_latex_symbols(strip_latex,
                                                                  latex_symbols=latex2gtd.ARRAY_SYMBOLS + latex2gtd.LATEX_SYMBOLS)
            refine_text = " ".join(latex_parts)
            texts.append(refine_text)

        for t_idx, text in enumerate(texts):
            tree_gt_list.append("".join([ann_core_name + '_crop_' + '{0:03d}'.format(t_idx), '\t', text + '\n']))

    with open(os.path.join(vars['total_cptn_path'], "total_caption.txt"), "w", encoding="utf8") as f:
        for i in range(len(tree_gt_list)):
            gt = tree_gt_list[i]
            f.write("{}".format(gt))

    # Match CRAFT TRAIN & TEST
    craft_train_list = sorted(utils.get_filenames(vars['craft_train_path'], extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(vars['craft_test_path'], extensions=utils.TEXT_EXTENSIONS))

    tree_train_list = []
    tree_test_list = []
    for tree_gt in tree_gt_list:
        gt_fname = tree_gt.split('\t')[0]
        gt_core_name = gt_fname.split('_crop')[0]
        tree_fname = gt_core_name + '.txt'
        match_train_fname = os.path.join(vars['craft_train_path'], 'gt_' + tree_fname)
        match_test_fname = os.path.join(vars['craft_test_path'], 'gt_' + tree_fname)

        if match_train_fname in craft_train_list:
            tree_train_list.append(tree_gt)
        elif match_test_fname in craft_test_list:
            tree_test_list.append(tree_gt)

    # Save train.txt file
    train_fpath = os.path.join(vars['train_cptn_path'], 'train_caption.txt')
    with open(train_fpath, 'w') as f:
        f.write(''.join(tree_train_list))

    test_fpath = os.path.join(vars['test_cptn_path'], 'test_caption.txt')
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
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    utils.folder_exists(vars['train_gtd_path'], create_=True)
    utils.folder_exists(vars['test_gtd_path'], create_=True)

    for tgt_mode in ['TRAIN', 'TEST']:
        if tgt_mode == 'TRAIN':
            latex_path = vars['train_cptn_path']
            gtd_path = vars['train_gtd_path']
        elif tgt_mode == 'TEST':
            latex_path = vars['test_cptn_path']
            gtd_path =  vars['test_gtd_path']

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

def main_crop(ini, common_info, logger=None):
    """
        {COMMON_DIR}의 이미지를 crop하여
        train_crop_path, test_crop_path에 저장한다.
    """

    # Init. path variables
    craft_train_path, craft_test_path = replace_string_from_dict(ini['craft_train_path'], common_info), replace_string_from_dict(ini['craft_test_path'], common_info)
    train_img_path, test_img_path = replace_string_from_dict(ini['train_img_path'], common_info), replace_string_from_dict(ini['test_img_path'], common_info)
    train_crop_path, test_crop_path = replace_string_from_dict(ini['train_crop_path'], common_info), replace_string_from_dict(ini['test_crop_path'], common_info)

    save_vars = {
        'train_img_path' : train_img_path, 'test_img_path' : test_img_path,
        'train_crop_path' : train_crop_path, 'test_crop_path' : test_crop_path,
    }

    craft_train_list = sorted(utils.get_filenames(craft_train_path, extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(craft_test_path, extensions=utils.TEXT_EXTENSIONS))
    logger.info(" [CRAFT-TRAIN GT] # Total gt number to be processed: {:d}.".format(len(craft_train_list)))

    for craft_list in [craft_train_list, craft_test_list]:
        if craft_list is craft_train_list:
            tgt_mode = 'TRAIN'
        elif craft_list is craft_test_list:
            tgt_mode = 'TEST'

        available_cpus = len(os.sched_getaffinity(0))
        mp_inputs = [(craft_fpath, save_vars, tgt_mode) for file_idx, craft_fpath in enumerate(craft_list)]

        # Multiprocess func.
        multi_process.run(func=load_craft_gt_and_save_crop_images, data=mp_inputs,
                          n_workers=available_cpus, n_tasks=len(craft_list), max_queue_size=len(craft_list), logger=logger)

    return True

def load_craft_gt_and_save_crop_images(craft_fpath, save_info, tar_mode, print_=False):
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
                raw_img_path = os.path.join(save_info['train_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(save_info['train_crop_path'],
                                         img_fname + '_crop_' + '{0:03d}'.format(tl_idx) + '.jpg')
            elif tar_mode == 'TEST':
                raw_img_path = os.path.join(save_info['test_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(save_info['test_crop_path'],
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

def main_merge(ini, common_info, logger=None):
    # Init. path variables
    global src_gt_path, dst_train_gt_path, dst_test_gt_path, src_crop_img_path, dst_crop_img_path, src_gtd_path, dst_gtd_path
    vars = init_merge_ini(ini, common_info)

    utils.folder_exists(vars['total_dataset_path'], create_=True)
    utils.folder_exists(vars['concat_dataset_path'], create_=True)


    datasets = [dataset for dataset in os.listdir(vars['dataset_path']) if dataset != 'total']
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    # Process total files
    train_gt_text_paths = []
    test_gt_text_paths = []
    total_gt_text_paths = []
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            # Replace {DIR_NAME}
            for key, val in vars.items():
                vars[key] = val.replace('{DIR_NAME}', dir_name)

            # Check folder exists
            if utils.folder_exists(vars['dst_train_crop_img_path']) and utils.folder_exists(vars['dst_test_crop_img_path']):
                logger.info(" # Already {} crop_img are exist".format(vars['total_dataset_path']))
            else:
                utils.folder_exists(vars['dst_train_crop_img_path'], create_=True), utils.folder_exists(vars['dst_test_crop_img_path'], create_=True)
            if utils.folder_exists(vars['dst_train_gtd_path']) and utils.folder_exists(vars['dst_test_gtd_path']):
                logger.info(" # Already {} gtd are exist".format(vars['total_dataset_path']))
            else:
                utils.folder_exists(vars['dst_train_gtd_path'], create_=True), utils.folder_exists(vars['dst_test_gtd_path'], create_=True)

            # 1) Apply symbolic link for gtd & img path
            # 2) Concat gt files
            for tgt_mode in ['TRAIN', 'TEST']:
                if tgt_mode is 'TRAIN':
                    src_gt_path, dst_gt_path = vars['src_train_gt_path'], vars['dst_train_gt_path']
                    src_crop_img_path, dst_crop_img_path = vars['src_train_crop_img_path'], vars['dst_train_crop_img_path']
                    src_gtd_path, dst_gtd_path = vars['src_train_gtd_path'], vars['dst_train_gtd_path']
                elif tgt_mode is 'TEST':
                    src_gt_path, dst_gt_path = vars['src_test_gt_path'], vars['dst_test_gt_path']
                    src_crop_img_path, dst_crop_img_path = vars['src_test_crop_img_path'], vars['dst_test_crop_img_path']
                    src_gtd_path, dst_gtd_path = vars['src_test_gtd_path'], vars['dst_test_gtd_path']

                # Sort & link img_path
                src_crop_imgs, dst_crop_imgs = sorted(utils.get_filenames(src_crop_img_path, extensions=utils.IMG_EXTENSIONS)), sorted(utils.get_filenames(dst_crop_img_path, extensions=utils.IMG_EXTENSIONS))
                src_gtds, dst_gtds = sorted(utils.get_filenames(src_gtd_path, extensions=['gtd'])), sorted(utils.get_filenames(dst_gtd_path, extensions=['gtd']))

                src_crop_fnames, dst_crop_fnames = [utils.split_fname(crop_img)[1] for crop_img in src_crop_imgs], [utils.split_fname(crop_img)[1] for crop_img in dst_crop_imgs]
                src_gtd_fnames, dst_gtd_fnames = [utils.split_fname(gtd)[1] for gtd in src_gtds], [utils.split_fname(gtd)[1] for gtd in dst_gtds]

                if any(src_fname not in dst_crop_fnames for src_fname in src_crop_fnames):
                    img_sym_cmd = 'find {} -name "*.jpg" -exec ln {} {} \;'.format(src_crop_img_path, '{}', dst_crop_img_path) # link each files
                    # img_sym_cmd = 'ln "{}"* "{}"'.format(src_crop_img_path, dst_crop_img_path)  # argument is long
                    subprocess.call(img_sym_cmd, shell=True)
                    logger.info(" # [Link {} img] files {} -> {}.".format(tgt_mode, src_crop_img_path, dst_crop_img_path))
                else:
                    logger.info(" # [Link {} img] files already generated : {}.".format(tgt_mode, dst_crop_img_path))
                if any(src_fname not in dst_gtd_fnames for src_fname in src_gtd_fnames):
                    gtd_sym_cmd = 'find {} -name "*.gtd" -exec ln {} {} \;'.format(src_gtd_path, '{}', dst_gtd_path) # link each files
                    # gtd_sym_cmd = 'ln "{}"* "{}"'.format(src_gtd_path, dst_gtd_path)  # argument is long
                    subprocess.call(gtd_sym_cmd, shell=True)
                    logger.info(" # [Link {} gtd] files {} -> {}.".format(tgt_mode, src_gtd_path, dst_gtd_path))
                else:
                    logger.info(" # [Link {} gtd] files already generated : {}.".format(tgt_mode, dst_gtd_path))

                # Add to list all label files
                if tgt_mode == 'TRAIN':
                    train_gt_text_paths.append(src_gt_path)
                    dst_train_gt_path = vars['dst_train_gt_path']
                    total_gt_text_paths.append(dst_train_gt_path)

                elif tgt_mode == 'TEST':
                    test_gt_text_paths.append(src_gt_path)
                    dst_test_gt_path = vars['dst_test_gt_path']
                    total_gt_text_paths.append(dst_test_gt_path)

        logger.info(" # Dst. train gt path : {}".format(dst_train_gt_path))
        logger.info(" # Dst. test gt path : {}".format(dst_test_gt_path))
        logger.info(" # Concat. gt path : {}".format(vars['total_gt_path']))

        # Merge all label files
        utils.concat_text_files(train_gt_text_paths, dst_train_gt_path)
        utils.concat_text_files(test_gt_text_paths, dst_test_gt_path)
        utils.concat_text_files(total_gt_text_paths, vars['total_gt_path'])

    return True

def init_merge_ini(ini, common_info):
    dataset_path = replace_string_from_dict(ini['dataset_path'], common_info)
    total_dataset_path = replace_string_from_dict(ini['total_dataset_path'], common_info)
    concat_dataset_path = replace_string_from_dict(ini['concat_dataset_path'], common_info)
    base_dir_name = common_info['base_dir_name']

    src_train_path, src_test_path = os.path.join(dataset_path, '{DIR_NAME}', 'train'), os.path.join(dataset_path, '{DIR_NAME}', 'test')
    dst_train_path, dst_test_path = os.path.join(total_dataset_path, 'train'), os.path.join(total_dataset_path, 'test')

    src_train_gt_path, src_test_gt_path = os.path.join(src_train_path, f'{base_dir_name}', 'train_caption.txt'), os.path.join(src_test_path, f'{base_dir_name}', 'test_caption.txt')
    dst_train_gt_path, dst_test_gt_path = os.path.join(dst_train_path, f'{base_dir_name}', 'train_caption.txt'), os.path.join(dst_test_path, f'{base_dir_name}', 'test_caption.txt')
    total_gt_path = os.path.join(concat_dataset_path, 'total_caption.txt')

    src_train_crop_img_path, src_test_crop_img_path = os.path.join(src_train_path, f'{base_dir_name}', 'crop_img/'), os.path.join(src_test_path, f'{base_dir_name}', 'crop_img/')
    dst_train_crop_img_path, dst_test_crop_img_path = os.path.join(dst_train_path, f'{base_dir_name}', 'crop_img/'), os.path.join(dst_test_path, f'{base_dir_name}', 'crop_img/')

    src_train_gtd_path, src_test_gtd_path = os.path.join(src_train_path, f'{base_dir_name}', 'gtd/'), os.path.join(src_test_path, f'{base_dir_name}', 'gtd/')
    dst_train_gtd_path, dst_test_gtd_path = os.path.join(dst_train_path, f'{base_dir_name}', 'gtd/'), os.path.join(dst_test_path, f'{base_dir_name}', 'gtd/')

    vars = {
        'dataset_path' : dataset_path, 'total_dataset_path' : total_dataset_path, 'concat_dataset_path' : concat_dataset_path, 'base_dir_name' : base_dir_name,
        'src_train_path' : src_train_path, 'src_test_path' : src_test_path, 'dst_train_path' : dst_train_path, 'dst_test_path' : dst_test_path,
        'src_train_gt_path': src_train_gt_path, 'src_test_gt_path': src_test_gt_path, 'dst_train_gt_path': dst_train_gt_path, 'dst_test_gt_path': dst_test_gt_path,
        'total_gt_path' : total_gt_path,
        'src_train_crop_img_path' : src_train_crop_img_path, 'src_test_crop_img_path' : src_test_crop_img_path, 'dst_train_crop_img_path' : dst_train_crop_img_path, 'dst_test_crop_img_path' : dst_test_crop_img_path,
        'src_train_gtd_path' : src_train_gtd_path, 'src_test_gtd_path' : src_test_gtd_path, 'dst_train_gtd_path' : dst_train_gtd_path, 'dst_test_gtd_path' : dst_test_gtd_path,
    }

    return vars

def main_generate_img_pkl(ini, common_info, logger=None):
    """
        cptn_path, crop_path 파일을 변환하여
        img_pkl_path에 저장한다.
    """

    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    dataset_type = common_info['dataset_type']
    if dataset_type == 'CROHME':
        dataset_path = '../data/CROHME/'
        vars['train_cptn_path'], vars['test_cptn_path'] = os.path.join(dataset_path, 'caption', 'train_caption.txt'), os.path.join(dataset_path, 'caption', 'test_caption.txt')
        vars['train_crop_path'], vars['test_crop_path'] = os.path.join(dataset_path, 'image', 'off_image_train/'), os.path.join(dataset_path, 'image', 'off_image_test/')
        vars['train_img_pkl_path'], vars['test_img_pkl_path'] = os.path.join(dataset_path, 'image','off_image_train.pkl'), os.path.join(dataset_path, 'image', 'off_image_test.pkl')

    for tgt_mode in ['TRAIN', 'TEST']:
        args = [
            '--dataset_type', dataset_type,
            '--op_mode', tgt_mode,
            '--cptn_path', vars[f'{tgt_mode.lower()}_cptn_path'],
            '--crop_path', vars[f'{tgt_mode.lower()}_crop_path'],
            '--img_pkl_path', vars[f'{tgt_mode.lower()}_img_pkl_path'],
        ]
        gen_pkl.main(gen_pkl.parse_arguments(args))

    return True

def main_generate_label_align_pkl(ini, common_info, logger=None):
    """
        gtd_path 파일을 변환하여
        label_pkl_path, align_pkl_path에 저장한다.
    """

    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    for tgt_mode in ['TRAIN', 'TEST']:
        args = [
            '--dataset_type', common_info['dataset_type'],
            '--op_mode', tgt_mode,
            '--gtd_path', vars[f'{tgt_mode.lower()}_gtd_path'],
            '--label_pkl_path', vars[f'{tgt_mode.lower()}_label_pkl_path'],
            '--align_pkl_path', vars[f'{tgt_mode.lower()}_align_pkl_path'],
        ]
        prepare_label.main(prepare_label.parse_arguments(args))

    return True

def main_generate_voc(ini, common_info, logger=None):
    """
        total_caption.txt 파일을 변환하여
        dictionary.txt에 저장한다.
    """

    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    args = [
        '--dataset_type', common_info['dataset_type'],
        '--total_cptn_path', vars['total_cptn_path'],
        '--dict_path', vars['dict_path'],
    ]
    gen_voc.main(gen_voc.parse_arguments(args))

    return True

def main_train(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    args = [
        '--dataset_type', common_info['dataset_type'],
        '--concat_dataset_path', vars['concat_dataset_path'],
        '--train_path', vars['train_path'],
        '--test_path', vars['test_path'],
        '--multi_gpu_flag', ini['multi_gpu_flag'],
        '--init_param_flag', ini['init_param_flag'],
        '--reload_flag', ini['reload_flag'],
        '--batch_size', ini['batch_size'],
        '--maxlen', ini['maxlen'],
        '--max_epochs', ini['max_epochs'],
        '--lrate', ini['lrate'],
        '--eps', ini['eps'],
        '--decay_c', ini['decay_c'],
        '--clip_c', ini['clip_c'],
        '--estop', ini['estop'],
        '--K', ini['K'],
        '--Kre', ini['Kre'],
    ]
    train_wap.main(train_wap.parse_arguments(args))

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
        main_crop(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'MERGE':
        main_merge(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'GENERATE_IMG_PKL':
        main_generate_img_pkl(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'GENERATE_LABEL_ALIGN_PKL':
        main_generate_label_align_pkl(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'GENERATE_VOC':
        main_generate_voc(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini[args.op_mode], common_info, logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['GENERATE_SPLIT_CPTN', 'GENERATE_GTD', 'CROP_IMG', 'MERGE', 'GENERATE_IMG_PKL', 'GENERATE_LABEL_ALIGN_PKL', 'GENERATE_VOC', 'TRAIN', 'TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'TRAIN' # GENERATE_SPLIT_CPTN / GENERATE_GTD / CROP_IMG / MERGE / GENERATE_IMG_PKL / GENERATE_LABEL_ALIGN_PKL / GENERATE_VOC / TRAIN / TEST
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


