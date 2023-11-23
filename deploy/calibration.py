import os
import argparse
import numpy as np
import torch
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
from pysot.datasets.dataset_deploy import TrkDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate HiFT Calibration Data')
    args = parser.parse_args()
    args.calib_z = "./calibration/template/"
    args.calib_x = "./calibration/search/"
    args.calib_e = "./calibration/embed/"
    args.calib_path = [args.calib_z, args.calib_x, args.calib_e]
    return args


def calibration():
    # preprocess and configure
    args = parse_args()
    for calib_path in args.calib_path:
        if not os.path.exists(calib_path): os.makedirs(calib_path)

    # build dataset
    train_dataset = TrkDataset()
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8,
                              pin_memory=True, sampler=None, drop_last=True)

    for iter_id, batch_data in enumerate(train_loader):
        template = batch_data['template']  # bx3x127x127
        search = batch_data['search']  # bx3x287x287
        embed = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        embed = embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        print('template shape: ', template.shape)
        print('search shape: ', search.shape)
        print('embed shape: ', embed.shape)

        if iter_id < 128:
            # z = np.transpose(template.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            z = template.numpy().astype(np.uint8)
            z.tofile(args.calib_path[0] + "z" + "_" + str(iter_id) + ".bin")
            # x = np.transpose(search.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            x = search.numpy().astype(np.uint8)
            x.tofile(args.calib_path[1] + "x" + "_" + str(iter_id) + ".bin")

            embed_ = embed.numpy().astype(np.float32)
            embed_.tofile(args.calib_path[2] + "e" + "_" + str(iter_id) + ".bin")

        else:
            break


if __name__ == '__main__':
    calibration()
