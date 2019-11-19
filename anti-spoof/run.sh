


#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network y2 --loss diregress --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network shuff --loss arcface --id 001 --per-batch-size 96

#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset faceid --network shuffse --loss arcface --id 001 --per-batch-size 96

rm -r models/shuffse-softmax-anti-003/
CUDA_VISIBLE_DEVICES='0,1' python -u train.py --dataset anti --network shuffse --loss softmax --lr 0.1 --id 003 --per-batch-size 32

#CUDA_VISIBLE_DEVICES='2' python -u test.py --data-dir ../datasets/anti-spoof --model 'models/shuffse-softmax-anti-001/model,2' 
