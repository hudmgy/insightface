
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --data-dir ../datasets/stranger --prefix model/20d-xx/model
#CUDA_VISIBLE_DEVICES='' python -u test.py --data-dir ../datasets/stranger --model 'model/20d-xx/model,2'

CUDA_VISIBLE_DEVICES='1,2' python -u train.py --data-dir ../datasets/stranger --prefix model/1d-100w/model --per-batch-size 512
