from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import time
from glob import glob
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

from pysot.core.config import cfg
from pysot.models.model_builder_X3 import ModelBuilder
from pysot.tracker.hift_tracker_X3 import HiFTTracker
from pysot.utils.deploy_helper import print_properties, bgr2nv12_opencv, get_frames


def parse_args():
    parser = argparse.ArgumentParser(description='HiFT demo')
    parser.add_argument('--config', type=str, default='../experiments/HiFT.yaml', help='config file')
    parser.add_argument('--inference', default='../bin/HiFT.bin', help='HiFT algorithm in Aerial Tracking algorithms')
    parser.add_argument('--video_name', default='../video/ChasingDrones', type=str, help='videos or image files')
    args = parser.parse_args()
    args.init_rect = [653, 221, 55, 40]  # for ../video/ChasingDrones
    return args


def main():

    print('===> load argparse and configs <====')
    args = parse_args()
    cfg.merge_from_file(args.config)

    print('===> load dnn models <====')
    inference = dnn.load(args.inference)
    dnn_model = {'inference': inference}
    print_properties(inference[0].inputs[0].properties)
    print_properties(inference[0].inputs[1].properties)
    print_properties(inference[0].outputs[0].properties)
    print_properties(inference[0].outputs[1].properties)
    print_properties(inference[0].outputs[2].properties)

    print('===> create inference model <====')
    net = ModelBuilder(model=dnn_model)

    print('===> build tracker <====')
    tracker = HiFTTracker(net)

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
