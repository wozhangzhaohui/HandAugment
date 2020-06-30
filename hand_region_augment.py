import cv2
import os, sys
import argparse
import numpy as np
from data.util import crop_hand
from data.hands19task1 import read_image, fx, fy, ppx, ppy, cube_len, joint_n


def hand_region_augment(args):
    joint_list_path = args.joint_list_path
    offset = args.offset
    img_base = args.img_base
    verbose = args.verbose
    dst_dir = args.dst_dir
    lines = [line.split() for line in open(joint_list_path, 'r').readlines()]
    file_list = [line[0] for line in lines]
    joint_list = [[float(x) for x in line[1:]] for line in lines]
    os.makedirs(dst_dir, exist_ok=True)

    frame_n = len(file_list)
    for fi, file, joint, in zip(range(frame_n), file_list, joint_list):
        if verbose > 0 and fi % 100 == 0:
            print('%.2f%% %d / %d' % (fi / frame_n * 100, fi, frame_n))
        img_depth = read_image(os.path.join(img_base, file))

        # crop hand
        joints3d_anno = np.asarray(joint.copy(), np.float32).reshape(joint_n, 3)
        img_hand = crop_hand(img_depth, joints3d_anno, ppx=ppx, ppy=ppy, fx=fx, fy=fy,
                             bbsize=cube_len * 2, center_joint=3, offset=offset, hand_thickness=20)
        dst_name = os.path.join(dst_dir, file)
        cv2.imwrite(dst_name, img_hand.astype(np.uint16))
    print('hand_region_augment done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand region augment')
    parser.add_argument('--joint_list_path', type=str, default='output/result/result.txt', help='Input model')
    parser.add_argument('--img_base', type=str, default='dataset/HANDS19_Challenge/Task1/test_images',
                        help='test set image folder')
    parser.add_argument('--dst_dir', type=str, default='cache/hands19task1/test_images_augment', help='out put folder')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--hand_thickness', type=int, default=20, help='hand thickness')
    parser.add_argument('--offset', type=int, default=30, help='distance of hand region expand')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    hand_region_augment(args)
