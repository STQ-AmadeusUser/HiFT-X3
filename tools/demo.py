from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
import time
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder_deploy import ModelBuilder
from pysot.tracker.hift_tracker_deploy import HiFTTracker
from pysot.utils.model_load import load_pretrain
from pysot.utils.deploy_helper import get_frames

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='HiFT demo')
parser.add_argument('--config', type=str, default='../experiments/HiFT.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='../snapshot/HiFT.pth', help='model name')
parser.add_argument('--video_name', default='../video/ChasingDrones', type=str, help='videos or image files')
args = parser.parse_args()
args.init_rect = [653, 221, 55, 40]  # for ../video/ChasingDrones


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    cfg.CUDA = False
    device = 'cpu'

    # create model
    model = ModelBuilder(device=device)

    # load model
    model = load_pretrain(model, args.snapshot, device=device).eval().to(device)

    # build tracker
    tracker = HiFTTracker(model)

    first_frame = True
    video_name = args.video_name.split('/')[-1].split('.')[0]

    for frame in get_frames(args.video_name):
        if first_frame:
            init_rect = args.init_rect
            tracker.init(frame, init_rect)
            first_frame = False
            # for video writer
            writer = cv2.VideoWriter('../video/' + video_name + '_result' + '.mp4',
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     30,
                                     (frame.shape[1], frame.shape[0]))
            bbox = list(map(int, init_rect))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
            writer.write(frame)
            time.sleep(2)

        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            writer.write(frame)
            time.sleep(2)


if __name__ == '__main__':
    main()
