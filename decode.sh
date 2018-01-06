CVDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/dev
CV_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/dev/ali.ark

EVALDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/eval
EVAL_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/eval/ali.ark



GPUS=`nvidia-smi|awk 'BEGIN{n=0}{if(NF==15){ print n,$11-$9;n=n+1  }}'| sort -k 2 -r|awk '{print $1}'| head -n 2`
gpus_str=`echo $GPUS`
gpus_list_str=`echo ${gpus_str[*]// /,}`

export CUDA_VISIBLE_DEVICES=${gpus_list_str}
export CUDA_LAUNCH_BLOCKING=1

/slwork/users/jqg01/tools/anaconda2/bin/python src/decode.py "$CVDATA" "$CV_LABELS" -o predict13 -m dev
/slwork/users/jqg01/tools/anaconda2/bin/python src/decode.py "$EVALDATA" "$EVAL_LABELS" -o predict14 -m eval
