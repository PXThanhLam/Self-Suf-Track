from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import sys
sys.path.append('/home/anhkhoa/Lam_working/human_tracking/FairMOT/src/lib')
import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 
import builtins
import torch.backends.cudnn as cudnn
from utils.utils import init_distributed_mode

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    
    print(opt)
    ###
    if opt.self_sup_aug :
        init_distributed_mode(opt)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    ###
    

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader
    if not opt.self_sup_aug :
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else :
        train_loader = torch.utils.data.DataLoader(
            dataset,
            sampler = sampler ,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )


    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    print('St epoch', start_epoch)
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        if opt.rank == 0:
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            
            # if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
            #             epoch, model, optimizer)
            # else:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                            epoch, model, optimizer)
                # save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                #         epoch, model, optimizer)
                # if epoch % 1 == 0 or epoch >= 25 :
                #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                #             epoch, model, optimizer)
            logger.write('\n')
        if epoch in opt.lr_step:
            if opt.rank == 0:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                param_group['lr'] = lr
        
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    opt.self_sup_aug = False
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    opt = opts().parse()
    main(opt)
   

# python -m torch.distributed.launch --nproc_per_node=4 train.py --exp_id crowdhuman_dla34 --gpus 0,1,2,3 --batch_size 8 --load_model '/home/anhkhoa/Lam_working/human_tracking/FairMOT/exp/mot/crowdhuman_dla34_ce_mosaic/model_42.pth' --num_epochs 100 --lr_step '60,80,90' --data_cfg '../src/lib/cfg/mot17.json' --lr 2e-4 
# python -m torch.distributed.launch --nproc_per_node=4 train.py --exp_id mot17_half_dla34_sim --gpus 0,1,2,3 --batch_size 7 --load_model '/home/anhkhoa/Lam_working/human_tracking/FairMOT/models/ctdet_coco_dla_2x.pth' --num_epochs 30 --lr_step '20' --data_cfg '../src/lib/cfg/mot17_half.json' --lr 2e-4