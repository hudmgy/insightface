


#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss diregress --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network shuff --loss arcface --id 001 --per-batch-size 96

rm -r models/shuffse-arcface-faceid-001
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset faceid --network shuffse --loss arcface --id 001 --per-batch-size 96
