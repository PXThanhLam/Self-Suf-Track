from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import math

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from models.model import create_model, load_model
from datasets.dataset.jde import letterbox
from models.utils import _tranpose_and_gather_feat
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process


def view_emb(
        opt,
        img_1_path,
        querry_pos,
        img_2_path,
        img_size=(1088, 608),
):
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()
    
    img1 = cv2.imread(img_1_path)  # BGR
    h, w, _ = img1.shape
    img1, ratio, padw, padh = letterbox(img1, height=img_size[1], width=img_size[0])
    img1_ori = img1.copy()
    img1 = img1[:, :, ::-1].transpose(2, 0, 1)
    img1 = np.ascontiguousarray(img1, dtype=np.float32)
    img1 /= 255.0

    img2 = cv2.imread(img_2_path)  # BGR
    img2, _, _, _ = letterbox(img2, height=img_size[1], width=img_size[0])
    img2_ori = img2.copy()
    img2 = img2[:, :, ::-1].transpose(2, 0, 1)
    img2 = np.ascontiguousarray(img2, dtype=np.float32)
    img2 /= 255.0

    querry_pos = np.array(querry_pos)
    querry_pos0 = querry_pos.copy()
    querry_pos[0] = ratio * w * (querry_pos0[0] - querry_pos0[2] / 2) + padw
    querry_pos[1] = ratio * h * (querry_pos0[1] - querry_pos0[3] / 2) + padh
    querry_pos[2] = ratio * w * (querry_pos0[0] + querry_pos0[2] / 2) + padw
    querry_pos[3] = ratio * h * (querry_pos0[1] + querry_pos0[3] / 2) + padh

    querry_point = ( (querry_pos[0] + querry_pos[2] ) / 2 , (querry_pos[1] + querry_pos[3] ) / 2 )

    embedding, id_labels = [], []
    print('Extracting features...')

    output1 = model(torch.tensor([img1]).cuda())[-1]
    querry_point = np.array(querry_point) // opt.down_ratio
    querry_point_ = int(img_size[0] // opt.down_ratio * querry_point[1] + querry_point[0]) 
    id_head1 = _tranpose_and_gather_feat(output1['id'], torch.tensor(np.array([[querry_point_]])).cuda())
    id_head1 = id_head1[0][0].cpu().data.numpy()
    id_head1 /= np.sqrt(np.sum(id_head1**2))

    id_head2 = model(torch.tensor([img2]).cuda())[-1]['id']
    id_head2 = id_head2[0].cpu().data.numpy().transpose(1,2,0)
    id_head2 /= np.sqrt(np.sum(id_head2**2,axis=2,keepdims=True))
    
    
    sim = np.einsum('i,mni->mn',id_head1,id_head2)
    print(np.sort(sim.flatten())[-100:])
    sim[sim<0] = 0
    sim[sim<0.84] /= 2
    print(np.sort(sim.flatten())[-100:])
    sim = cv2.resize(sim,img_size)


    querry_pos = np.array(querry_pos, dtype = np.int32)
    cv2.rectangle(img1_ori,(querry_pos[0], querry_pos[1]), (querry_pos[2], querry_pos[3]),(0,0,255), 1)
    cv2.imwrite('query.png',img1_ori)
    sim =  np.repeat(sim[:, :, np.newaxis], 3, axis=2) * 255
    sim = np.array(sim,np.float32)
    img2_ori = np.array(img2_ori,np.float32)

    img2_ori = cv2.addWeighted(img2_ori,0.5, sim, 0.5,0)
    cv2.imwrite('res.png', img2_ori )

    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    opt.self_sup_aug = False
    view_emb(
        opt,
        img_1_path = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17/images/train/MOT17-02-SDP/img1/000505.jpg',
        querry_pos = (0.473438, 0.516667, 0.043750, 0.277778),
        img_2_path = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17/images/train/MOT17-02-SDP/img1/000595.jpg',
        img_size=(1088, 608),
)
    