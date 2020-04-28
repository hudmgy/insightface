export PATH="/usr/local/bin:/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"
export INCLUDE="/usr/local/include:/usr/local/cuda-9.0/include:$INCLUDE"


CUDA_VISIBLE_DEVICES='1' python -u test.py --data-dir ../datasets/anti-spoof --model 'models/r34-arcface-emore-001/model,100' --test-batch-size 128
