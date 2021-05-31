import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy

import sys
sys.path.append('/home/anhkhoa/Lam_working/human_tracking/FairMOT/src/lib')
sys.path = sys.path[1:]
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, color_aug, random_crop
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta,xyxy2stdxywh, list_mapping
import time
from tqdm import tqdm


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608), video_w = 1920, video_h = 1080):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = video_w, video_h
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        # img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, ds):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        mosaic = random.random() < self.mosaic_prob
        if mosaic:
            img, labels = load_mosaic(self, img_path, label_path, ds)
        augment_hsv = True
        # if self.augment and augment_hsv:
        #     # SV augmentation by 50%
        #     fraction = 0.5
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #     S = img_hsv[:, :, 1].astype(np.float32)
        #     V = img_hsv[:, :, 2].astype(np.float32)

        #     a = (random.random() * 2 - 1) * fraction + 1
        #     S *= a
        #     if a > 1:
        #         np.clip(S, a_min=0, a_max=255, out=S)

        #     a = (random.random() * 2 - 1) * fraction + 1
        #     V *= a
        #     if a > 1:
        #         np.clip(V, a_min=0, a_max=255, out=V)

        #     img_hsv[:, :, 1] = S.astype(np.uint8)
        #     img_hsv[:, :, 2] = V.astype(np.uint8)
        #     cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        img_ori = img.copy()
        if self.color_aug :
            img = np.float32(img) / 255
            color_aug(self._data_rng, img, self._eig_val, self._eig_vec)
            img = np.clip(img * 255 , 0, 255)
        img = np.array(img, np.float32)
        h, w, _ = img.shape
        
        ##COPY PASTE
        if os.path.isfile(label_path):
            labels0 = labels if mosaic else np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
        else:
            labels0 = []
        labels_ori = labels0.copy()
        img_original = img.copy()
        copy_paste = random.random() < self.copypaste_prob
        if copy_paste and len(labels0) > 0:
            img, labels0 = load_copy_paste(self, img_original , labels0, vis_thres = 0.6, keep_prob = 0.99, update_anno = True)
        ####

        # adjust label and resize img
        img, ratio, padw, padh = letterbox(img, height=height, width=width)
        if os.path.isfile(label_path):
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
                           
        else:
            labels = np.array([])

                        
        # Augment image and labels        
        if self.augment: 
            img, labels, M,_ = random_affine(img, labels)
            
        #### Self-sup augment
        if self.self_sup_aug :
            img_ori = np.float32(img_ori) / 255
            color_aug(self._data_rng, img_ori, self._eig_val, self._eig_vec)
            img_ori = np.clip(img_ori * 255 , 0, 255)
            img_ori = np.array(img_ori, np.float32)
            if random.random() > 0.05 :
                k_size_h = int(0.003 * height * random.random())
                k_size_w = int(0.003 * width * random.random())
                img_ori = cv2.GaussianBlur(img_ori,(2 * k_size_h + 1 ,2 * k_size_w + 1),cv2.BORDER_DEFAULT)

            img_aug1, labels_aug1 = load_copy_paste(self, img_ori , labels_ori , vis_thres = 0.55, keep_prob = 0.99, update_anno = False)
            # Normalized xywh to pixel xyxy format
            img_aug1, ratio, padw, padh = letterbox(img_aug1, height=height, width=width)
            labels_aug = labels_aug1.copy()
            labels_aug[:, 2] = ratio * w * (labels_aug1[:, 2] - labels_aug1[:, 4] / 2) + padw
            labels_aug[:, 3] = ratio * h * (labels_aug1[:, 3] - labels_aug1[:, 5] / 2) + padh
            labels_aug[:, 4] = ratio * w * (labels_aug1[:, 2] + labels_aug1[:, 4] / 2) + padw
            labels_aug[:, 5] = ratio * h * (labels_aug1[:, 3] + labels_aug1[:, 5] / 2) + padh
            labels_aug1 = labels_aug

            im_aug1, labels_aug1,_, clip_ori_ratio = random_affine(img_aug1,labels_aug1, degrees=(-7, 7), translate= [(-0.2,0.2),(-0.2,0.2)],  scale=(0.4, 1.2), shear=(-4, 4)) 
            im_mean = np.array((np.mean(im_aug1[:,:,0]),np.mean(im_aug1[:,:,1]),np.mean(im_aug1[:,:,2])))
            # random dropout :
            # if random.random() >= 0.1:
            #     drop_idx = (labels_aug1[:,6]) >= 0.6
            #     for idx,taken in enumerate(drop_idx) :
            #         if not taken :
            #             continue
            #         x1,y1,x2,y2 = np.array(labels_aug1[idx][2:6], dtype = np.int32)
            #         rand_w = int(np.random.uniform(0.2, 0.7) * (x2 - x1))
            #         rand_h = int(np.random.uniform(0.2, 0.7) * (y2 - y1))
            #         try:
            #             x = random.randint(x1, x2  - rand_w)
            #             y = random.randint(y1, y2  - rand_h)
            #             #im_aug1[y1:y2,x1:x2,:] = cv2.resize(im_aug1[y:y+rand_h, x:x+rand_w,:],((x2 - x1), (y2 - y1)))
            #             im_aug1[y:y+rand_h, x:x+rand_w,:] = np.zeros((rand_h, rand_w,3)) + im_mean
            #         except:
            #             pass
            ###
            nL = len(labels_aug1)
            if nL > 0:
                labels_aug1[:, 2:6] = xyxy2stdxywh(labels_aug1[:, 2:6].copy(),width,height) 
            if (random.random() > 0.5 ):
                im_aug1 = np.fliplr(im_aug1)
                if nL > 0:
                    labels_aug1[:, 2] = 1 - labels_aug1[:, 2]
            labels_aug1 = labels_aug1[clip_ori_ratio >=0.1]
            im_aug1 = np.ascontiguousarray(im_aug1[:, :, ::-1])  # BGR to RGB
            if self.transforms is not None:
                im_aug1 = self.transforms(im_aug1)           
        
        ##############
        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2stdxywh(labels[:, 2:6].copy(),width,height)  # / height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)
        if self.self_sup_aug:
            return img,labels, img_path, (h,w), im_aug1, labels_aug1, mosaic
        else:
            return img, labels, img_path, (h, w), mosaic

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-4, 4), translate= [(-0.1,0.1),(-0.1,0.1)],  scale=(0.8, 1.1), shear=(-1.5, 1.5),
                  borderValue=(127.5, 127.5, 127.5), border = (0,0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    height = int(img.shape[0] + border[1] * 2)
    width = int(img.shape[1] + border[0] * 2)

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(translate[0][0], translate[0][1]) * width   # x translation (pixels)
    T[1, 2] = random.uniform(translate[1][0], translate[1][1])* height   # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T
            xy = (xy[:, :2]).reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
            

            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            # Remove out of img and small box
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10) & ( xy[:,0] < width) & ( xy[:,1] < height) & ( xy[:,2] >= 0) & ( xy[:,3] >= 0)
            targets = targets[i]
            targets[:, 2:6] = xy[i]
            # Adjust the visibility
            x = targets[:,[2,4]]
            x_clip = np.clip(x,0,width) 
            y = targets[:,[3,5]]
            y_clip = np.clip(y,0,height)
            area_clip = (x_clip[:,1] - x_clip[:,0])*(y_clip[:,1] - y_clip[:,0])
            area_ori = (x[:,1] - x[:,0]) * (y[:,1] - y[:,0])
            clip_ori_ratio = area_clip / area_ori
            targets[:,6] *= np.clip(clip_ori_ratio  ,0, 1)

        return imw, targets, M , clip_ori_ratio
    else:
        return imw


def load_image(self, img_path):
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    r = min(self.width/w0, self.height/h0)
    if r != 1:  
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2] 

def xywhn2xyxy(x, w=1088, h=608, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def point_in_rect(x1,y1, rec_w, rec_h):
    return (0 < x1 and x1 < rec_w) and (0 < y1 and y1 < rec_h)

def load_mosaic(self, ori_img_path,ori_label_path, ori_ds):
    labels4 = []
    s = (self.width, self.height)
    xc, yc = [int(random.uniform(-x, 2 * s[i] + x)) for i,x in enumerate(self.mosaic_border)]  # mosaic center x, y
    np.random.seed()
    indices = ['_'] + list(np.random.randint(0,self.nF, size=3))  # 3 additional image indices
    pad_w_left  = []
    pad_h_up  = []
    pad_w_right = []
    pad_h_down = []
    for i, index in enumerate(indices):
        if i != 0 :
            for j, c in enumerate(self.cds):
                if index >= c:
                    ds = list(self.label_files.keys())[j]
                    start_index = c
            img_path = self.img_files[ds][index - start_index]
            label_path = self.label_files[ds][index - start_index]
        else:
            ds = ori_ds
            img_path = ori_img_path
            label_path = ori_label_path

        img, _, (h, w) = load_image(self, img_path)
       
        if i == 0:  # top left
            img4 = np.full((s[1] * 2, s[0] * 2, img.shape[2]), 127.5, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[0] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[0] * 2), min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        if i == 0:
            pad_w_left.append(padw)
            pad_h_up.append(padh)
        if i == 1:
            pad_h_up.append(padh)
            pad_w_right.append(padw + x2a - x1a)
        if i == 2:
            pad_h_down.append(padh + y2a - y1a)
            pad_w_left.append(padw)
        if i == 3:
            pad_h_down.append(padh + y2a -y1a)
            pad_w_right.append(padw + x2a-x1a)
               

        # Labels
        labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
        if labels.size:
            labels[:, 2:6] = xywhn2xyxy(labels[:, 2:6], w, h, padw, padh)  # normalized xywh to pixel xyxy format     

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]   
        labels4.append(labels)

    
    pad_w_left, pad_w_right, pad_h_down, pad_h_up = max(min(pad_w_left),0), min(max(pad_w_right),2*s[0]), min(max(pad_h_down), 2*s[1]),max(min(pad_h_up),0)

    if pad_w_right - pad_w_left > self.width and pad_h_down - pad_h_up > self.height :
        pad_w_left = np.random.randint(pad_w_left, pad_w_left + (pad_w_right - pad_w_left - self.width)//2)
        pad_w_right = np.random.randint(pad_w_right - (pad_w_right - pad_w_left - self.width)//2, pad_w_right)
        pad_h_up = np.random.randint(pad_h_up, pad_h_up + (pad_h_down - pad_h_up - self.height)//2)
        pad_h_down = np.random.randint(pad_h_down - (pad_h_down - pad_h_up - self.height)//2 , pad_h_down)

    labels4 = np.concatenate(labels4, 0)
    img4 = img4[pad_h_up:pad_h_down,pad_w_left:pad_w_right]
    labels4[:, [2,4]] -= pad_w_left
    labels4[:, [3,5]] -= pad_h_up

    # #######################
    # im_w, im_h = pad_w_right - pad_w_left, pad_h_down - pad_h_up
    # img4_raw  = np.full((s[1] * 3, s[0] * 3, img.shape[2]), 127.5, dtype=np.uint8)
    # img4_raw[s[1]:im_h + s[1], s[0]:im_w + s[0]] = img4
    # img4_fill = img4_raw.copy()
    # draw_name = str(np.random.randint(1,25))
    # for fbox in labels4:
    #     fbox = fbox[2:6]
    #     x1,y1,x2,y2 = np.array(fbox, np.int32) + np.array([s[0],s[1],s[0],s[1]],  np.int32)
    #     color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    #     cv2.rectangle(img4_raw, (x1,y1), (x2,y2),color, 3) 
    # cv2.imwrite('/home/anhkhoa/Lam_working/human_tracking/data_exp/ima_raw' + draw_name + '.jpg',np.uint8(img4_raw))
    #####################
    

    # delete all detect outsise image
    
    im_w, im_h = pad_w_right - pad_w_left, pad_h_down - pad_h_up
    x1 = labels4[:,2]
    y1 = labels4[:,3]
    x2 = labels4[:,4]
    y2 = labels4[:,5]
    valid_point = (x1 < im_w - 5 ) & (x2 > 5 ) & (y1 < im_h - 5 ) & (y2 > 5 )
    valid_point = np.array(valid_point)
    labels4 = labels4[valid_point]

    # ###################
    # for fbox in labels4:
    #     fbox = fbox[2:6]
    #     x1,y1,x2,y2 = np.array(fbox, np.int32) + np.array([s[0],s[1],s[0],s[1]],  np.int32)
    #     color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    #     cv2.rectangle(img4_fill, (x1,y1), (x2,y2),color, 3) 
    # cv2.imwrite('/home/anhkhoa/Lam_working/human_tracking/data_exp/ima_back' + draw_name + '.jpg',np.uint8(img4_fill))    

    ####################

    # convert to normalize xywh
    top_l = labels4[:,3]/ ( pad_h_down - pad_h_up)
    x = ((labels4[:,2]  + labels4[:,4])/2) 
    y = ((labels4[:,3]  + labels4[:,5])/2) 
    w = (labels4[:,4]  - labels4[:,2]) 
    h = (labels4[:,5]  - labels4[:,3]) 
    
    labels4[:,2] = x / (pad_w_right - pad_w_left)
    labels4[:,3] = y / ( pad_h_down - pad_h_up)
    labels4[:,4] = w / (pad_w_right - pad_w_left)
    labels4[:,5] = h / ( pad_h_down - pad_h_up)
    
    # i = np.random.randint(10)
    # print(ori_img_path)
    # cv2.imwrite('/home/anhkhoa/Lam_working/human_tracking/data_exp/' + str(i) + '.png', img4)
    # print(ori_label_path)
    # print(labels4)
    return img4, labels4

def copy_paste(image_patch, paste_img, paste_mask):
    return image_patch * (1 - paste_mask) + paste_img * paste_mask
def load_copy_paste(self, original_image , anno, vis_thres = 0.6, update_anno = True, keep_prob = 0.8) :
    if keep_prob is None :
        keep_prob = self.copy_paste_prob
    all_patch = []
    anno = list(anno)
    for bbox in anno:
        x,y,w,h,vis = bbox[2:]
        if vis >= vis_thres :
            t,l,b,r = x - w/2, y - h/2, x + w/2, y + h/2
            h, w, _ = original_image.shape
            t , l , b , r = t * w ,l * h , b * w,r * h 
            all_patch.append([t, l, b, r])

    im_h, im_w, _ = original_image.shape
    rand_idxs = np.random.randint(0, len(self.copy_paste_crop_path) , max(2 * len(all_patch) // 3, 2))
    rand_crop_path = list(self.copy_paste_crop_path[rand_idxs])
    rand_crop_path_mask = list(self.copy_paste_mask_path[rand_idxs])

    rand_crop_id = self.copy_paste_id[rand_idxs]
    rand_crop_img = [cv2.imread(path) for path in rand_crop_path]
    rand_crop_mask = [cv2.imread(path) for path in rand_crop_path_mask]
    rand_crop_img_size = [ img.shape[:2] for img in rand_crop_img]
    for box in all_patch:
        t ,l, b, r = box
        box_w, box_h = b - t, r - l
        matches_score = []
        for rand_crop_h,rand_crop_w  in rand_crop_img_size :
            match_score = (min(rand_crop_w,box_w)/max(rand_crop_w,box_w) + min(rand_crop_h,box_h)/max(rand_crop_h,box_h)) / 2
            matches_score.append(match_score)
        if len(matches_score) < 1 :
            continue
        match_idx = np.argmax(matches_score)
        if matches_score[match_idx] < 0.75 :
            continue

        blend_img = rand_crop_img[match_idx]
        blend_img_h, blend_img_w, _= blend_img.shape
        blend_mask = rand_crop_mask[match_idx]

        del rand_crop_img[match_idx]
        del rand_crop_img_size[match_idx]
        del rand_crop_mask[match_idx]
        if t < 0 or l < 0 or b > im_w or r > im_h :
            continue
        left_or_right = np.random.randint(0,2) > 0
        if left_or_right :
            rand_blend_t = np.random.randint(int(t - 0.8 * blend_img_w), int(t - 0.2 * blend_img_w))
            rand_blend_l = np.random.randint(int(l - 0.1 * blend_img_h), int(l + 0.1* blend_img_h ))
            rand_blend_b = rand_blend_t + blend_img_w
            rand_blend_r = rand_blend_l + blend_img_h
        else :
            rand_blend_t = np.random.randint(int(t +  0.4 * blend_img_w), int(t + 1.2 * blend_img_w ))
            rand_blend_l = np.random.randint(int(l - 0.1 * blend_img_h), int(l + 0.1* blend_img_h ))
            rand_blend_b = rand_blend_t + blend_img_w
            rand_blend_r = rand_blend_l + blend_img_h

        if rand_blend_t > im_w - 100 or rand_blend_l > im_h - 100 or rand_blend_b < 100 or rand_blend_r < 100 :
            continue
        if random.random() > keep_prob :
            continue
        if min(rand_blend_r,im_h) - max(rand_blend_l,0) < 0 or min(rand_blend_b,im_w) - max(rand_blend_t,0) < 0 :
            continue
        if blend_mask is None :
            print('errr')
            continue
        blend_mask = np.array(blend_mask) / 255
        blend_img = blend_img[0 : min(rand_blend_r,im_h) - max(rand_blend_l,0), 0 : min(rand_blend_b,im_w) - max(rand_blend_t,0),:]
        blend_mask = blend_mask[0 : min(rand_blend_r,im_h) - max(rand_blend_l,0), 0 : min(rand_blend_b,im_w) - max(rand_blend_t,0),:]
        original_image[ max(rand_blend_l,0) : min(rand_blend_r,im_h), max(rand_blend_t,0) : min(rand_blend_b,im_w), :] = \
        copy_paste(original_image[ max(rand_blend_l,0) : min(rand_blend_r,im_h), max(rand_blend_t,0) : min(rand_blend_b,im_w), :], blend_img, blend_mask)
        if update_anno :
            anno.append([0,rand_crop_id[match_idx],(rand_blend_t + rand_blend_b) / 2 / im_w,(rand_blend_l + rand_blend_r) / 2 / im_h, blend_img_w / im_w, blend_img_h / im_h,1])

    return original_image, np.array(anno)
def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 7)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
            ], dtype=np.float32)
        self._data_rng = np.random.RandomState(123)
        self.color_aug = not opt.no_color_aug
        self.mosaic_border = [- 2 * img_size[0] // 3, - 2 * img_size[1] // 3]
        self.mosaic_prob = opt.mosaic_prob
        self.mix_prob = opt.mixup_prob
        for ds, path in tqdm(paths.items()):
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                # x.replace('images_', 'labels_with_ids/').replace('.png', '.txt').replace('.jpg', '.txt')
                # x.replace('images_', 'labels_with_ids/').replace('.png', '.txt').replace('.jpg', '.txt')
                # if 'CrowdHuman' in x else x.replace('images', 'labels_with_ids_agnous').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds] ]

        for ds, label_paths in tqdm(self.label_files.items()):
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v
        ###### For copy paste#######
        print(self.tid_start_index)
        self.copypaste_prob = opt.copypaste_prob
        mask_roots = ['/home/anhkhoa/Lam_working/human_tracking/mot_data/CrowdHuman/images_train_mask',
                      '/home/anhkhoa/Lam_working/human_tracking/mot_data/CrowdHuman/images_val_mask'] 
        self.copy_paste_id = []
        self.copy_paste_crop_path = []
        self.copy_paste_mask_path = []
        i = 1000000
        for idx, root in enumerate(mask_roots) :
            ds = 'crowdhuman_train' if idx == 0 else 'crowdhuman_test'
            for mask_id in tqdm(os.listdir(root + '/crop_box' )) :
                self.copy_paste_crop_path.append(root + '/crop_box/' + mask_id)
                self.copy_paste_mask_path.append(root + '/crop_mask/' + mask_id)
                # self.copy_paste_id.append(np.loadtxt(root + '/crop_id/' + mask_id.split('.')[0] + '.txt', dtype=np.float64, delimiter=',') + self.tid_start_index[ds])
                self.copy_paste_id.append(i)
                i += 1
        self.copy_paste_crop_path = np.array(self.copy_paste_crop_path)
        self.copy_paste_mask_path = np.array(self.copy_paste_mask_path)
        self.copy_paste_id = np.array( self.copy_paste_id)
        #############################
        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        self.self_sup_aug = opt.self_sup_aug
        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('total # image:', self.nF)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if self.self_sup_aug:
            imgs, labels, img_path, (input_h, input_w), img_aug, labels_aug, from_mosiac = self.get_data(img_path, label_path, ds)
        else:
            imgs, labels, img_path, (input_h, input_w), from_mosiac = self.get_data(img_path, label_path, ds)

        if not from_mosiac :
            for i, _ in enumerate(labels):
                if labels[i, 1] > -1:
                    labels[i, 1] += self.tid_start_index[ds]
            if self.self_sup_aug :
                for i, _ in enumerate(labels_aug):
                    if labels_aug[i, 1] > -1:
                        labels_aug[i, 1] += self.tid_start_index[ds]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = min(labels.shape[0], self.max_objs)
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)
        visibility = np.zeros((self.max_objs, ), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        if self.self_sup_aug:
            num_objs_aug = min(labels_aug.shape[0],self.max_objs)
            hm_aug  = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
            ids_aug = np.zeros((self.max_objs, ), dtype=np.int64)
            ind_aug = np.zeros((self.max_objs, ), dtype=np.int64)
            reg_mask_aug = np.zeros((self.max_objs, ), dtype=np.uint8)
            bbox_xys_aug = np.zeros((self.max_objs, 4), dtype=np.float32)

        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:6]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w  
            bbox[[1, 3]] = bbox[[1, 3]] * output_h 
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]

            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)

            h = bbox[3]
            w = bbox[2]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_amodal
                visibility[k] = label[6]
            
           
        if self.self_sup_aug:
            for k in range(num_objs_aug) :
                label = labels_aug[k]
                bbox = label[2:6]
                cls_id = int(label[0])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w  
                bbox[[1, 3]] = bbox[[1, 3]] * output_h

                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]

                bbox[0] = np.clip(bbox[0], 0, output_w - 1)
                bbox[1] = np.clip(bbox[1], 0, output_h - 1)
                h = bbox[3]
                w = bbox[2] 
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    radius = 6 if self.opt.mse_loss else radius
                    ct = np.array(
                        [bbox[0], bbox[1]], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(hm_aug[cls_id], ct_int, radius)
                    ind_aug[k] = ct_int[1] * output_w + ct_int[0]
                    ids_aug[k] = label[1]
                    reg_mask_aug[k] = 1
                    bbox_xys_aug[k] = bbox_amodal

        if not self.self_sup_aug:
            ret = {'input': imgs / 255, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys, 'visibility' : visibility}
        else:
            ret = {'input': imgs / 255, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys,
                     'visibility' : visibility,'input_aug':img_aug / 255, 'hm_aug' : hm_aug, 'reg_mask_aug': reg_mask_aug, 'ind_aug': ind_aug, 'ids_aug' : ids_aug, 'bbox_aug' : bbox_xys_aug}
        return ret


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.mosaic_prob = 0.0
        self.self_sup_aug = False 
        self.color_aug = False
        self.copypaste_prob = 0.0
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('total # image:', self.nF)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)

        imgs, labels, img_path, (h, w), _  = self.get_data(img_path, label_path, ds)

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs/ 255 , labels0, img_path, (h, w)


if __name__ == '__main__' :
    opt = opts().parse()
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size= 2,
        shuffle=True,
        num_workers= 8,
        pin_memory=True,
        drop_last=True
    )
    idx = 1

    start = time.time()
    for batch in iter(train_loader) :
        im = cv2.cvtColor(batch['input'][0].permute(1,2,0).cpu().data.numpy()* 255, cv2.COLOR_RGB2BGR)
        bboxs = batch['bbox'][0][batch['reg_mask'][0] > 0]
        heat = batch['hm'][0][0].cpu().data.numpy()* 255
        heat =  np.repeat(heat[:, :, np.newaxis], 3, axis=2)
        visibility = batch['visibility'][0].cpu().data.numpy()
        # im = im + 0.8 * cv2.resize(heat,(1260, 708))

        id_   = batch['ids'][0][batch['reg_mask'][0] > 0].cpu().data.numpy()
        id_aug = batch['ids_aug'][0][batch['reg_mask_aug'][0] > 0].cpu().data.numpy()

        
        mapping = list_mapping(id_,id_aug)
        map_id_ = np.array(mapping)[:,0]
        map_id_aug = np.array(mapping)[:,1]
        
        id_ = id_[map_id_]
        id_aug = id_aug[map_id_aug]

        for i,(fbox,vis) in enumerate(zip(bboxs[map_id_],visibility[map_id_])):
            x1,y1,x2,y2 = np.array(fbox, np.int32) * opt.down_ratio 
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            cv2.rectangle(im, (x1,y1), (x2,y2),color, 2) 
            cv2.putText(im, str(id_[i]), ((x1 + x2)//2, (y1 + y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite('/home/anhkhoa/Lam_working/human_tracking/data_exp/im_' + str(idx) + '.jpg',np.uint8(im))


        im_aug = cv2.cvtColor(batch['input_aug'][0].permute(1,2,0).cpu().data.numpy()* 255, cv2.COLOR_RGB2BGR)
        bboxs_aug = batch['bbox_aug'][0][batch['reg_mask_aug'][0] > 0]
        heat_aug = batch['hm_aug'][0][0].cpu().data.numpy()* 255
        heat_aug =  np.repeat(heat[:, :, np.newaxis], 3, axis=2)
        # im_aug = im_aug + 0.8 * cv2.resize(heat,(1260, 708))
        for i,fbox in enumerate(bboxs_aug[map_id_aug]):
            x1,y1,x2,y2 = np.array(fbox, np.int32) * opt.down_ratio 
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            cv2.rectangle(im_aug, (x1,y1), (x2,y2),color, 2) 
            cv2.putText(im_aug, str(id_aug[i]), ((x1 + x2)//2, (y1 + y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite('/home/anhkhoa/Lam_working/human_tracking/data_exp/im_aug_' + str(idx) + '.jpg',np.uint8(im_aug))
                
        idx +=1
        if idx == 15:
            break
    print((time.time() -start) / 320 )
        