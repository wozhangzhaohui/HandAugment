import cv2
import numpy as np


def crop_image(img, center, cube_len, fx, fy, width, height):
    bb2d = [center[0] - cube_len / center[2] * fx,
            center[1] - cube_len / center[2] * fy,
            center[0] + cube_len / center[2] * fx,
            center[1] + cube_len / center[2] * fy]
    src = np.float32([[bb2d[0], bb2d[1]], [bb2d[2], bb2d[1]], [bb2d[0], bb2d[3]]])
    dst = np.float32([[0, 0], [width, 0], [0, height]])
    trans = cv2.getAffineTransform(src, dst)
    crop = 0
    crop = cv2.warpAffine(img, trans, (width, height), crop, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT,
                          center[2] + cube_len)
    crop[crop == 0] = center[2] + cube_len
    crop -= center[2]
    crop[crop > cube_len] = cube_len
    crop[crop < -cube_len] = -cube_len
    crop /= cube_len
    return crop


def crop_hand(img, skeleton, ppx=315.944855, ppy=245.287079, fx=475.065948, fy=475.065857, bbsize=300.0, center_joint=3,
              offset=30, hand_thickness=20):
    minu, maxu = min(skeleton[:, 0]) - offset, max(skeleton[:, 0]) + offset
    minv, maxv = min(skeleton[:, 1]) - offset, max(skeleton[:, 1]) + offset
    mind, maxd = min(skeleton[:, 2]) - offset, max(skeleton[:, 2]) + offset
    mind -= hand_thickness

    height = img.shape[0]
    width = img.shape[1]

    # create point cloud and mask with minu/v/d and maxu/v/d
    rows = np.repeat((np.arange(height) + 1).reshape((1, height)).T, width, axis=1)
    columns = np.repeat((np.arange(width) + 1).reshape((1, width)), height, axis=0)

    xyz = pixel2world(columns, rows, (img + 1).copy(), ppx, ppy, fx, fy)
    gtx, gty, gtz = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    mask1 = np.zeros((height, width))
    mask2 = np.zeros((height, width))
    mask3 = np.zeros((height, width))

    mask1[np.logical_and(gtx >= minu, gtx <= maxu)] = 1
    mask2[np.logical_and(gty >= minv, gty <= maxv)] = 1
    mask3[np.logical_and(gtz >= mind, gtz <= maxd)] = 1
    mask = np.logical_and(np.logical_and(mask1, mask2), mask3)

    img_hand = img.copy()
    img_hand[mask==0] = 0
    return img_hand

def pixel2world(u, v, d, ppx, ppy, fx, fy):
    x = ((u - ppx) * d) / fx
    y = ((v - ppy) * d) / fy
    return np.concatenate([x[:, np.newaxis], y[:, np.newaxis], d[:, np.newaxis]], axis=1)

def world2pixel(x, y, z, ppx, ppy, fx, fy):
    u = ((x * fx) / z) + ppx
    v = ((y * fy) / z) + ppy
    return np.concatenate([u[:, np.newaxis], v[:, np.newaxis], z[:, np.newaxis]], axis=1)

def camera2uvd(joint, fx, fy, ppx, ppy):
    joint_shape = joint.shape
    joint = np.asarray(joint, dtype=np.float32).reshape(-1, 3)
    joint[:, 0] = joint[:, 0] / joint[:, 2] * fx + ppx
    joint[:, 1] = joint[:, 1] / joint[:, 2] * fy + ppy
    joint = joint.reshape(joint_shape)
    return joint


def uvd2norm(joint, center, fx, fy, cube_len):
    for j in range(int(len(joint) / 3)):
        joint[j * 3 + 0] = (joint[3 * j + 0] - center[0]) / (cube_len / center[2] * fx)
        joint[j * 3 + 1] = (joint[3 * j + 1] - center[1]) / (cube_len / center[2] * fy)
        joint[j * 3 + 2] = (joint[3 * j + 2] - center[2]) / cube_len
    return joint


def norm2world(joint, center, fx, fy, ppx, ppy, cube_len):
    joint[:, 0] = joint[:, 0] * fx * cube_len / center[:, 2] + center[:, 0]
    joint[:, 1] = joint[:, 1] * fy * cube_len / center[:, 2] + center[:, 1]
    joint[:, 2] = joint[:, 2] * cube_len + center[:, 2]
    joint[:, 0] = (joint[:, 0] - ppx) / fx * joint[:, 2]
    joint[:, 1] = (joint[:, 1] - ppy) / fy * joint[:, 2]
    return joint