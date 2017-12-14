#!/usr/bin/env bash

<<EOF
KALDI_HOME="/home/slhome/jqg01/work-home/tools/kaldi-trunk"
export PATH=$PATH:${KALDI_HOME}/src/featbin/
export PATH=$PATH:${KALDI_HOME}/src/gmmbin/
export PATH=$PATH:${KALDI_HOME}/src/bin
export KALDI_ROOT=/home/slhome/jqg01/work-home/tools/kaldi-trunk


BASE=`pwd`
mfcc=$BASE/mfcc
fbank=$BASE/fbank

DELTAS="add-deltas ark:- ark:-"
FEXT="splice-feats --left-context=11 --right-context=11 ark:- ark:-"

TRAINDATA="copy-feats scp:$fbank/train/fbank.scp ark:- | $DELTAS | $FEXT |"
CVDATA="copy-feats scp:$fbank/dev/fbank.scp ark:- |  $DELTAS | $FEXT |"

TR_COUNTS=$fbank/train/counts.ark
CV_COUNTS=$fbank/dev/counts.ark
EOF



TR_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/train/ali.ark
CV_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/dev/ali.ark


TRAINDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/train
CVDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/dev


GPUS=`nvidia-smi|awk 'BEGIN{n=0}{if(NF==15){ print n,$11-$9;n=n+1  }}'| sort -k 2 -r|awk '{print $1}'| head -n 2`
gpus_str=`echo $GPUS`
gpus_list_str=`echo ${gpus_str[*]// /,}`

export CUDA_VISIBLE_DEVICES=${gpus_list_str}
export CUDA_LAUNCH_BLOCKING=1

/slwork/users/jqg01/tools/anaconda2/bin/python src/train.py "$TRAINDATA" "$TR_LABELS" "$CVDATA" "$CV_LABELS"
#"$TR_COUNTS" "$CV_COUNTS"
#/slwork/users/jqg01/tools/anaconda2/bin/python src/train.py "$TR_LABELS" "$CV_LABELS"
#python src/train.py "$TRAINDATA" "$TR_LABELS" "$CVDATA" "$CV_LABELS"
