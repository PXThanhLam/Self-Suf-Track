import sys
sys.path.append('/home/anhkhoa/Lam_working/human_tracking/FairMOT/src/lib')
from lib.tracking_utils.evaluation import Evaluator
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import logging
import os
import os.path as osp
import motmetrics as mm
import numpy as np
#run twostage)demo to get result file, then run this
#file to get evaluate result
#'MOT16-02','MOT16-04','MOT16-09','MOT16-10','MOT16-11','MOT16-13',
#'MOT16-05','MOT16-10','MOT16-11','MOT16-13'
data_root = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17/images/results/mot17_train_self'
gt_root = '/home/anhkhoa/Lam_working/human_tracking/mot_data/MOT17/images/train/'
seqs = ('MOT17-02-SDP', 'MOT17-04-SDP','MOT17-05-SDP','MOT17-09-SDP','MOT17-10-SDP','MOT17-11-SDP','MOT17-13-SDP')
# seqs = ('MOT17-05-SDP','MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP')
# seqs = ('MOT17-02-SDP', 'MOT17-04-SDP','MOT17-09-SDP')
def main( data_root=data_root, seqs=seqs, exp_name='demo'):
    logger.setLevel(logging.INFO)
    
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
       
        result_filename = os.path.join(data_root, '{}.txt'.format(seq))
        print(result_filename)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(gt_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        
  

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

main()