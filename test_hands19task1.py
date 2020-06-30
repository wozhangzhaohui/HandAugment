import os, sys
import argparse
import torch
from data.util import norm2world
from data.hands19task1 import Hands19Task1TestDataset, get_center_from_bbx, fx, fy, ppx, ppy, cube_len, joint_n


def unit_test(args):
    model_path = args.model_path
    output_result_path = args.output_result_path
    input_test_img_folder = args.input_test_img_folder
    verbose = args.verbose
    batch_size = args.batch_size

    print('load model')
    net = torch.load(model_path)
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    net = net.to('cuda').eval()
    net = torch.nn.DataParallel(net)

    print('prepare test center')
    get_center_from_bbx(bb_path='dataset/HANDS19_Challenge/Task1/test_bbs.txt',
                        dst_path='cache/hands19task1/test_center_uvd.txt',
                        bbx_rectify=True)
    center_list_path = 'cache/hands19task1/test_center_uvd.txt'

    print('start test')
    file_name_list = [line.split()[0] for line in open(center_list_path, 'r').readlines()]
    test_dataset = Hands19Task1TestDataset(center_list_path, input_test_img_folder)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8, pin_memory=False)
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    fo = open(output_result_path, 'w')
    fi = 0
    frame_n = len(file_name_list)
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            if verbose > 0:
                print('%.2f%% %d / %d' % (fi / frame_n * 100, fi, frame_n))
            (inputs, param) = [i.to('cuda') for i in sample]
            outputs = net(inputs)
            j_pd = outputs.reshape([-1, 3])
            c = param.reshape([-1, 1, 3]).expand([-1, joint_n, -1]).reshape([-1, 3])
            j_pd = norm2world(j_pd, c, fx, fy, ppx, ppy, cube_len)
            j_pd = j_pd.reshape([-1, joint_n * 3]).tolist()
            for i in range(len(j_pd)):
                if param.reshape([-1, 3])[i][2].item() < 0.1:
                    joint = ['\t0' for x in range(joint_n * 3)]
                else:
                    joint = ['\t' + '%.4f' % (x) for x in j_pd[i]]
                fo.write(file_name_list[fi])
                fo.writelines(joint)
                fo.write('\n')
                fi += 1
    if verbose > 0:
        print("test done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get result.txt.')
    parser.add_argument('-m', '--model_path', type=str,
                        default='/mnt/data1/zzh/zzh_local_new/hands19/task1/models/stage0_mano_5m/model_2_ep080.pth',
                        help='Input model')
    parser.add_argument('-o', '--output_result_path', type=str, default='result/result.txt',
                        help='Output path of result.txt')
    parser.add_argument('-i', '--input_test_img_folder', type=str,
                        default='dataset/HANDS19_Challenge/Task1/test_images',
                        help='test set image folder')
    parser.add_argument('--gpu_id', type=str, default='0', help='use specific gpu')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    unit_test(args)
