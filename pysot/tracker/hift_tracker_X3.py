from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import cv2
from pysot.core.config import cfg


class HiFTTracker(object):
    def __init__(self, model):
        super(HiFTTracker, self).__init__()

        self.score_size = cfg.TRAIN.OUTPUT_SIZE
        self.anchor_num = 1
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.model = model
        self.generate_grids()

    @staticmethod
    def dcon(x):
        x[np.where(x <= -1)] = -0.99
        x[np.where(x >= 1)] = 0.99
        return (np.log(1 + x) - np.log(1 - x)) / 2

    def generate_grids(self):
        size = cfg.TRAIN.OUTPUT_SIZE
        self.grid_x = np.tile(
            (cfg.ANCHOR.STRIDE * (np.linspace(0, size-1, size)) + 63) - cfg.TRAIN.SEARCH_SIZE//2, size
        ).reshape(-1)
        self.grid_y = np.tile(
            (cfg.ANCHOR.STRIDE * (np.linspace(0, size-1, size)) + 63).reshape(-1, 1) - cfg.TRAIN.SEARCH_SIZE//2, size
        ).reshape(-1)

        self.grid_xx = np.int16(np.tile(np.linspace(0, size-1, size), size).reshape(-1))
        self.grid_yy = np.int16(np.tile(np.linspace(0, size-1, size).reshape(-1, 1), size).reshape(-1))

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.image = img
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.firstbbox = np.concatenate((self.center_pos,self.size))

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scaleaa = s_z

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)  # 1x127x127x3
        self.template = z_crop
        self.model.template(z_crop)

    def con(self, x):
        return x * (cfg.TRAIN.SEARCH_SIZE//2)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scaleaa
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z

        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)  # 1x287x287x3

        outputs = self.model.track(x_crop)

        # generate_anchor
        shap = (self.dcon(outputs['loc'].squeeze())) * 143
        w = shap[0, self.grid_yy, self.grid_xx] + shap[1, self.grid_yy, self.grid_xx]
        h = shap[2, self.grid_yy, self.grid_xx] + shap[3, self.grid_yy, self.grid_xx]
        x = self.grid_x - shap[0, self.grid_yy, self.grid_xx] + w/2
        y = self.grid_y - shap[2, self.grid_yy, self.grid_xx] + h/2
        anchor = np.zeros((cfg.TRAIN.OUTPUT_SIZE ** 2, 4))
        anchor[:, 0] = x
        anchor[:, 1] = y
        anchor[:, 2] = w
        anchor[:, 3] = h
        pred_bbox = anchor.transpose()

        score1 = outputs['cls1'].flatten() * cfg.TRACK.w2
        score2 = outputs['cls2'].flatten() * cfg.TRACK.w3
        score = (score1 + score2) / 2

        def change(r):
            return np.maximum(r, 1. / (r+1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/(self.size[1]+1e-5)) /
                     (pred_bbox[2, :]/(pred_bbox[3, :]+1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        
        bbox = pred_bbox[:, best_idx] / scale_z
        
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR 

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score,
               }

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch[np.newaxis, :, :, :]

        return im_patch
