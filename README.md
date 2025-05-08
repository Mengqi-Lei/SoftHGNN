# SoftHGNN
Implementation of the paper "SoftHGNN: Soft Hypergraph Neural Networks for General Visual Recognition".
# Getting Started🚀
## Classification
## CrowdCouting
### 1. Data Preparation
#### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
        |->gt_density_maps/
        |    |->train/
        |    |->test/  
```
### 2. Training
The network can be trained using the train.py script. For training on ShanghaiTech PartA, use
```
python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./runs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
## ObjectDetection
