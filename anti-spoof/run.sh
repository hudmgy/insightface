
rm -r models/shuffse-softmax-anti-003/
CUDA_VISIBLE_DEVICES='0,1' python -u train.py --dataset anti --network shuffse --loss softmax --lr 0.1 --id 003 --per-batch-size 32

