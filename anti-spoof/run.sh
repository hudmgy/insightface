export PATH="/usr/local/bin:/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"
export INCLUDE="/usr/local/include:/usr/local/cuda-9.0/include:$INCLUDE"


#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset anti --network shuffse --loss softmax --lr 0.0003 --id 011 --per-batch-size 64 --pretrained models/shuffse-arcface-emore-002/model --pretrained-epoch 3

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset anti --network y2 --loss softmax --id 001 --per-batch-size 256

CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train.py --dataset fqua --network y2 --loss softmax --lr 0.0003 --id 002 --per-batch-size 128
