#!/usr/bin/python
import os
import cv2
import torch
import torch.utils.data
import numpy as np
from .util import crop_image

img_w = 640
img_h = 480
fx = 475.065948
fy = 475.065857
ppx = 315.944855
ppy = 245.287079
cube_len = 150
channel = 1
joint_n = 21


def read_image(img_path):
    img_depth = cv2.imread(img_path, 2).astype(np.float32)
    return img_depth


class Hands19Task1TestDataset(torch.utils.data.Dataset):
    def __init__(self, center_list_path, img_base, crop_width=224, crop_height=224):
        lines = [line.split() for line in open(center_list_path, 'r').readlines()]
        self.path_list = [os.path.join(img_base, line[0]) for line in lines]
        self.center_list = [[float(x) for x in line[1:]] for line in lines]
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __getitem__(self, index):
        img_path = self.path_list[index]
        img_depth = self._read_image(img_path)
        center = self.center_list[index]
        if center[2] == 0:
            data = torch.FloatTensor(channel, self.crop_height, self.crop_width)
            crop_param = np.array(center, dtype=np.float32)
            return data, crop_param
        img_crop = crop_image(img_depth, center, cube_len, fx, fy, self.crop_width, self.crop_height)
        data = torch.from_numpy(
            np.asarray(img_crop, dtype=np.float32).reshape(channel, self.crop_height, self.crop_width))
        crop_param = np.array(center, dtype=np.float32)
        return data, crop_param

    def __len__(self):
        return len(self.path_list)

    def _read_image(self, img_path):
        return read_image(img_path)


def get_center_from_bbx(bb_path='dataset/HANDS19_Challenge/Task1/test_bbs.txt',
                        dst_path='cache/hands19task1/test_center_uvd.txt',
                        bbx_rectify=True):
    lines = [line.split() for line in open(bb_path).readlines()]
    file_list = [line[0] for line in lines]
    bb_list = [[int(x) for x in line[1:]] for line in lines]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fo = open(dst_path, 'w')
    for bb, file in zip(bb_list, file_list):
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        ww = max(w, h)
        if bbx_rectify:
            bb_edge_status = [bb[0] == 0,
                              bb[1] == 0,
                              bb[2] == img_w,
                              bb[3] == img_h]
            edge_cnt = 0
            for i in range(4):
                if bb_edge_status[i]:
                    edge_cnt += 1

            if edge_cnt == 1:
                if (bb_edge_status[0]):
                    bb[0] = bb[2] - ww
                elif (bb_edge_status[1]):
                    bb[1] = bb[3] - ww
                elif (bb_edge_status[2]):
                    bb[2] = bb[0] + ww
                elif (bb_edge_status[3]):
                    bb[3] = bb[1] + ww

        center_uvd = [(bb[0] + bb[2]) / 2,
                      (bb[1] + bb[3]) / 2,
                      cube_len * 2 / ww * fx]
        center_str = ['\t' + '%.4f' % (x) for x in center_uvd]
        fo.write(file)
        fo.writelines(center_str)
        fo.write('\n')

