CVDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/dev
CV_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/dev/ali.ark

EVALDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/cqcc/eval
EVAL_LABELS=/home/slhome/jqg01/work-home/workspace/asvspoof2017/ali/eval/ali.ark


EVALDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/fft/eval/fft.scp
CVDATA=/home/slhome/jqg01/work-home/workspace/asvspoof2017/fft/dev/fft.scp


GPUS=`nvidia-smi|awk 'BEGIN{n=0}{if(NF==15){ print n,$11-$9;n=n+1  }}'| sort -k 2 -r|awk '{print $1}'| head -n 2`
gpus_str=`echo $GPUS`
gpus_list_str=`echo ${gpus_str[*]// /,}`

export CUDA_VISIBLE_DEVICES=${gpus_list_str}
export CUDA_LAUNCH_BLOCKING=1

#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$CVDATA" "$CV_LABELS" -o predict15 --mode "dev" --model "cnn_cqcc"
#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$EVALDATA" "$EVAL_LABELS" -o predict16 --mode "eval" --model "cnn_cqcc"

#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$CVDATA" "$CV_LABELS" --model "resnet18_cqcc" -o predict13 --mode "dev"
#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py  -o predict14 --model "resnet18_cqcc"  --mode "eval" "$EVALDATA" "$EVAL_LABELS"
#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$CVDATA" "$CV_LABELS" -o predict17 --mode "dev" --model "resnet18_fft"
#/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$EVALDATA" "$EVAL_LABELS" -o predict18 --mode "eval" --model "resnet18_fft"

/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$CVDATA" "$CV_LABELS" -o predict19 --mode "dev" --model "cnn_fft"
/slwork/users/jqg01/tools/anaconda2/bin/python src/decode_new.py "$EVALDATA" "$EVAL_LABELS" -o predict20 --mode "eval" --model "cnn_fft"

