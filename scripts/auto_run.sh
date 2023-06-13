CONFIG=./configs/COCO-InstanceSegmentation/BoxSnake_RCNN/boxsnake_R_50_FPN_1x.yaml
python train_net.py --num-gpus 8 --config-file $CONFIG --resume # 'resume' is convenient for the elastic devices
