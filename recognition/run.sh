export PATH="/usr/local/bin:/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"
export INCLUDE="/usr/local/include:/usr/local/cuda-9.0/include:$INCLUDE"


#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss diregress --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network shuff --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset faceid --network shuffse --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset emore --network shuff --loss arcface --id 001 --per-batch-size 128 


#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train.py --dataset covered --network r50 --loss arcface --id 001 --per-batch-size 128

#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train.py --dataset covered --network r100 --loss arcface --id 001 --per-batch-size 128


#CUDA_VISIBLE_DEVICES='1' python -u train.py --dataset covered --network r100 --pretrained models/r100-arcface-covered-001/model --pretrained-epoch 156 --lr 0.00001 --loss arcface --id 0011 --per-batch-size 32

#CUDA_VISIBLE_DEVICES='1' python -u train.py --dataset covered --network r100 --loss arcface --id 002 --per-batch-size 32 

#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train.py --dataset emore --network r34 --loss arcface --id 001 --per-batch-size 64

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train.py --dataset emore --network varg --lr 0.05 --loss arcface --id 002 --per-batch-size 64
