import os
import numpy as np
from tqdm import tqdm

def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)
root = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17_test/labels_with_ids'
label_root = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17_test/labels_with_ids_agnous/test'
mkdirs(label_root)
tid_curr = 1
for txt_file in os.listdir(root) :
    gt = np.loadtxt(os.path.join(root,txt_file), dtype=np.float64, delimiter=',')
    seq = txt_file.split('.')[0]
    seq_label_root = os.path.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    seq_info = open(os.path.join('/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17_test/images/test', seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
    for fid, _ , x, y, w, h, _ , _ , _ , _ in tqdm(gt):
        fid = int(fid)
        x += w / 2
        y += h / 2
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height , 1)
        label_fpath = os.path.join(seq_label_root, '{:06d}.txt'.format(fid))
        with open(label_fpath, 'a') as f:
            f.write(label_str)
        tid_curr += 1

