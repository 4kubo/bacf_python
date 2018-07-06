import cv2
import os
import re
import json
from glob import glob
from operator import itemgetter
from numpy import array

path_to_config_files = "config"


def get_sequence_info(paths_to_seq, seq_names, test_seq_names, dataset_names=None):
    # Load config information for exceptional files
    path_to_json = "{0}/seq_info_otb.json".format(path_to_config_files)
    with open(path_to_json, "r") as f:
        config_dict = json.load(f)

    # At first, collect information for all sequences
    _total_seq_names = seq_names + test_seq_names
    total_seq_names, _frames_names, path_to_each_gt =\
        _get_path_to_seq(_total_seq_names, paths_to_seq, config_dict)

    # Ground-truth labels and frame names of the target sequence
    _gt_labels = _get_gt_bbox(path_to_each_gt)

    invalid_flag = False
    for i, (gt_label, frame_names) in enumerate(zip(_gt_labels, _frames_names)):
        if len(gt_label) != len(frame_names):
            print("Sequence {} is invalid".format(total_seq_names[i]))
            invalid_flag = True

    if invalid_flag:
        raise Exception("There are invalid seqs")

    _n_frames = [len(e) for e in _gt_labels]

    # Exclude sequences which overlap with training ones
    target_idx, test_idx = _exclude(total_seq_names, seq_names, test_seq_names)

    seq_names = [total_seq_names[i] for i in target_idx]
    gt_labels = [_gt_labels[i] for i in target_idx]
    frames_names = [_frames_names[i] for i in target_idx]
    n_frames = [_n_frames[i] for i in target_idx]

    test_seq_names = [total_seq_names[i] for i in test_idx]
    test_gt_labels = [_gt_labels[i] for i in test_idx]
    test_frames_names = [_frames_names[i] for i in test_idx]
    test_n_frames = [_n_frames[i] for i in test_idx]

    info = [seq_names, gt_labels, frames_names, n_frames]
    test_info = [test_seq_names, test_gt_labels, test_frames_names, test_n_frames]

    return info, test_info

def _get_path_to_seq(seq_names, paths_to_seq, config_dict):
    """
    Args:
        seq_names: The list of names of sequence
        paths_to_seq: Absolute path to data-set
        config_dict:

    Returns:

    """
    frames_names = [[]] * len(seq_names)
    path_to_each_gt = [""] * len(seq_names)

    # Pair of dirs under `path_to_seq` and `path_to_seq`
    seq_of_each_path = []
    for path_to_seq in paths_to_seq:
        if os.path.exists(path_to_seq):
            seqs = glob("{0}/*".format(path_to_seq))
            seqs = [seq.split("/")[-1] for seq in seqs]
            seq_of_each_path.append([seqs, path_to_seq])

    for seqs, path_to_seq_ in seq_of_each_path:
        for i, seq_name in enumerate(seq_names):
            # The case that the sequence has multi target in OTB dataset
            if "-" in seq_name and seq_name in config_dict["Multi"]:
                num = config_dict["Multi"][seq_name]
                groundtruth_file_name = 'groundtruth_rect.{0}.txt'.format(num)
                seq_name = seq_name.split("-")[0]
            else:
                groundtruth_file_name = 'groundtruth_rect.txt'

            # The case that `seq_name` exists under the `path_to_seq` directory
            # else skip
            if seq_name in seqs and len(frames_names[i]) is 0:
                # A list of frame names
                frame_names = _get_frame_names(path_to_seq_, seq_name)
                path = "{0}/{1}/{2}".format(path_to_seq_, seq_name, groundtruth_file_name)
                if os.path.exists(path):
                    # OTB
                    if seq_name in config_dict["Exceptional length"]:
                        tartet_dict = config_dict["Exceptional length"][seq_name]
                        start = tartet_dict["start"]
                        end = tartet_dict["end"]
                    else:
                        start = 0
                        end = len(frame_names)
                else :
                    # Temple-color-128
                    groundtruth_file_name = '{0}_gt.txt'.format(seq_name)

                    path = "{0}/{1}/{1}_frames.txt".format(path_to_seq_, seq_name)
                    with open(path, "r") as f:
                        line = f.readline()
                    split_line = re.split(r',|\tc|\s', line.strip())
                    start = int(split_line[0])-1
                    end = int(split_line[1])
                frame_names = frame_names[start: end]

                path_to_gt = "{0}/{1}/{2}".format(path_to_seq_, seq_name, groundtruth_file_name)
                assert os.path.exists(path_to_gt)
                path_to_each_gt[i] = path_to_gt
                frames_names[i] = frame_names

    # Check about invalid sequences input
    invalid_seq_index1 = [i for i, element in enumerate(path_to_each_gt) if len(element) is 0]
    invalid_seq_index2 = [i for i, element in enumerate(frames_names) if len(element) is 0]
    assert invalid_seq_index1 == invalid_seq_index2
    if len(invalid_seq_index1) is not 0:
        raise Exception("Invalid seqs : {0}"
                        .format(", ".join(seq for seq in itemgetter(*invalid_seq_index1)(seq_names))))

    # Arange valid values
    valid_seq_index = [i for i, element in enumerate(path_to_each_gt) if len(element) is not 0]
    valid_seq_names = [seq_names[i] for i in valid_seq_index]
    valid_frames_names = [frames_names[i] for i in valid_seq_index]
    valid_path_to_each_gt = [path_to_each_gt[i] for i in valid_seq_index]
    return valid_seq_names, valid_frames_names, valid_path_to_each_gt

def _get_frame_names(path_to_seq_, seq_name):
    frame_names = glob("{0}/{1}/img/*".format(path_to_seq_, seq_name))
    frame_names = sorted(frame_names, key=_numerical_sort)
    return frame_names

def _numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def _get_gt_bbox(path_to_each_gt):
    gt_labels = [[]]*len(path_to_each_gt)
    for i, path_to_gt in enumerate(path_to_each_gt):
        if os.path.exists(path_to_gt):
            with open(path_to_gt, "r") as f:
                bboxes = [[int(float(w)) for w in re.split(r',|\tc|\s', line.strip())]
                          for line in f.readlines()]
            gt_labels[i] = array(bboxes)

    return gt_labels

def _exclude(total_seq_names, seq_names, test_seq_names):
    """
    Arrange target sequence index and test sequence index. Test sequences take priority
    Args:
        total_seq_names:
        seq_names:
        test_seq_names:

    Returns:

    """
    assert ([seq_name in total_seq_names for seq_name in seq_names])
    assert ([test_seq_name in total_seq_names for test_seq_name in seq_names])
    target_idx, test_idx = [], []
    temp_seq_names, temp_test_seq_names = [], []
    for i, seq_name in enumerate(total_seq_names):
        if seq_name in test_seq_names:
            if not seq_name in temp_test_seq_names:
                test_idx.append(i)
                temp_test_seq_names.append(seq_name)
        elif seq_name in seq_names:
            if not seq_name in temp_seq_names:
                target_idx.append(i)
                temp_seq_names.append(seq_name)
        else:
            raise IndexError("Sequence {0} does not exist".format(seq_name))
    return target_idx, test_idx

def load_image(image_names):
    images = [cv2.imread(image_name) for image_name in image_names]
    return images