import os
from tqdm import tqdm
root = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17_test/images/test'
write = '/home/anhkhoa/Lam_working/human_tracking/FairMOT/src/data/mot17.test'
file = open(write,'w')
for f1 in os.listdir(root) :
    for f2 in tqdm(os.listdir(root + '/' + f1 + '/' + 'img1')) :
        file.write('MOT17_test/images/test/' + f1 + '/' + 'img1/' + f2 + '\n')

