#!/bin/bash

TR_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/train/ali.ark
CV_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/dev/ali.ark


TRAINDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/mfcc/train/mfcc.scp
CVDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/mfcc/dev/mfcc.scp


GPUS=`nvidia-smi|awk 'BEGIN{n=0}{if(NF==15){ print n,$11-$9;n=n+1  }}'| sort -k 2 -n -r|awk '{print $1nvidia-smi|awk 'BEGIN{n=0}{if(NF==15){ print n,$11-$9;n=n+1  }}'| sort -k 2 -r|awk '{print $21}'| head -n 2`
gpus_str=`echo $GPUS`
gpus_list_str=`echo ${gpus_str[*]// /,}`

export CUDA_VISIBLE_DEVICES=${gpus_list_str}
export CUDA_LAUNCH_BLOCKING=1

/slwork/users/jqg01/tools/anaconda2/bin/python src/train_new.py "$TRAINDATA" "$TR_LABELS" "$CVDATA" "$CV_LABELS" -model cnn_mfcc

