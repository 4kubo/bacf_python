import re
from glob import glob
import cv2
from numpy import array

groundtruth_file_name = 'groundtruth_rect.txt'


def get_sequence_info(path_to_sequences, target_sequence_name):
    # Ground-truth
    target_gt = path_to_sequences + '/' + target_sequence_name + '/' + groundtruth_file_name
    gt_label = get_gt_bbox(target_gt)

    frame_names = glob(path_to_sequences + '/' + target_sequence_name + '/img/*')
    frame_names = sorted(frame_names, key=numerical_sort)

    n_frames = len(frame_names)

    init_rect = gt_label[0, :]

    return gt_label, frame_names, n_frames, init_rect

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_gt_bbox(gt_file_name):
    with open(gt_file_name) as f:
        bboxes = [[int(w) for w in re.split(r',|\tc|\s', line.strip())]
        for line in f.readlines()]
    return array(bboxes)

def load_image(image_names):
    images = [cv2.imread(image_name) for image_name in image_names]
    return images