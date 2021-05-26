from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss, NT_Xent
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer
from utils.utils import list_mapping


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        if self.opt.ce_loss:
            self.classifier = nn.Linear(self.emb_dim, self.nID)
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        if self.opt.simclr_loss :
            self.sim_clr_predictor = nn.Sequential(nn.Linear(opt.reid_dim, opt.reid_dim), nn.ReLU(), nn.Linear(opt.reid_dim, self.opt.simclr_out_dim))
            self.sim_clr_loss = NT_Xent(self.opt.simclr_temp, self.opt.world_size)

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        # tuple mean feeding augment image (i.e using self-sup loss)
        if isinstance(outputs, tuple) :
            outputs, outputs_aug = outputs

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_target = batch['ids'][batch['reg_mask'] > 0]
                if self.opt.ce_loss :
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_output = self.classifier(id_head).contiguous()
                    id_loss += self.IDLoss(id_output, id_target)
                if self.opt.self_sup_aug and self.opt.simclr_loss :
                    id_head_aug = _tranpose_and_gather_feat(outputs_aug[s]['id'], batch['ind_aug'])
                    id_head_aug = id_head_aug[batch['reg_mask_aug'] > 0].contiguous()
                    id_target_aug = batch['ids_aug'][batch['reg_mask_aug'] > 0]
                    
                    id_target_numpy = id_target.cpu().data.numpy()
                    id_target_aug_numpy = id_target_aug.cpu().data.numpy()
                    idx_pair = np.array(list_mapping(id_target_numpy,id_target_aug_numpy))

                    id_head_pred = self.sim_clr_predictor(id_head[idx_pair[:,0]][:1500])
                    id_head_aug_pred = self.sim_clr_predictor(id_head_aug[idx_pair[:,1]][:1500])
                    
                    id_loss += self.sim_clr_loss(id_head_pred,id_head_aug_pred)
                   

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
