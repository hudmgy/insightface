MODEL=/home/chai/code/insightface/recognition/models/y2-arcface-retina/model


INPUT=/home/chai/Data/iccv19-challenge/iccv19-challenge-data
OUTPUT=/home/chai/code/insightface/recognition/models/y2-arcface-retina/image_y2_arc.bin
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u gen_image_feature.py --batch_size 128 --input $INPUT --output $OUTPUT --model $MODEL,1


INPUT=/home/chai/Data/iccv19-challenge/iQIYI-VID-FACE
OUTPUT=/home/chai/code/insightface/recognition/models/y2-arcface-retina/video_y2_arc.bin
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u gen_video_feature.py --batch_size 128 --input $INPUT --output $OUTPUT --model $MODEL,1
