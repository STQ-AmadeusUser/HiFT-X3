import os
import glob
import cv2
import yaml
import numpy as np


def get_frames(video_name):
    if video_name == '':
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


# bgr格式图片转换成 NV12格式
def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def center_image(img, screen_width=1920, screen_height=1080):
    if len(img.shape) == 2:
        imgT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        imgT = img
    irows, icols = imgT.shape[0:2]
    scale_w = screen_width * 1.0 / icols
    scale_h = screen_height * 1.0 / irows
    final_scale = min([scale_h, scale_w])
    final_rows = int(irows * final_scale)
    final_cols = int(icols * final_scale)
    print(final_rows, final_cols)
    imgT = cv2.resize(imgT, (final_cols, final_rows))
    diff_rows = screen_height - final_rows
    diff_cols = screen_width - final_cols
    img_show = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img_show[diff_rows // 2:(diff_rows // 2 + final_rows), diff_cols // 2:(diff_cols // 2 + final_cols), :] = imgT
    return img_show
