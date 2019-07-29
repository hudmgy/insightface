#MODEL=/home/chai/code/insightface/recognition/models/y2-diregress-retina-001/model
#MODEL=/home/chai/code/insightface/recognition/models/y2-arcface-retina-001/model
MODEL=/home/chai/code/insightface/recognition/models/shuff-arcface-retina-001/model



INPUT=/home/chai/Data/iccv19-challenge/iQIYI-VID-FACE
OUTPUT=/home/chai/code/insightface/recognition/models/shuff-arcface-retina-001/iQIYI-light
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u gen_video_feature.py --batch_size 32 --input $INPUT --output $OUTPUT --model $MODEL,1339


INPUT=/home/chai/Data/iccv19-challenge/iccv19-challenge-data
OUTPUT=/home/chai/code/insightface/recognition/models/shuff-arcface-retina-001/deepglint-light
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u gen_image_feature.py --batch_size 32 --input $INPUT --output $OUTPUT --model $MODEL,1339

