# SoftHGNN
Implementation of the paper "SoftHGNN: Soft Hypergraph Neural Networks for General Visual Recognition".
# Getting Started🚀
## Classification
## CrowdCouting
### 1. Data Preparation
#### Dataset structures:
```
DATA_ROOT/
        |->train_data/
        |    |->images/
        |    |    |->IMG_1.jpg
        |    |    |->IMG_2.jpg
        |    |    |->...
        |    |->ground_truth/
        |    |    |->GT_IMG_1.mat
        |    |    |->GT_IMG_2.mat
        |    |    |->... 
        |->test_data/  
```
### 2. Training
The network can be trained using the train.py script. For training on ShanghaiTech PartA with using 'Pyramid ViT' backbbone and 'SoftHGNN-SeS' module, use
```
python train.py --data-dir $DATA_ROOT \
    --dataset_file 'sha' \
    --lr 0.00001\
    --max-epochs 4000 \
    --val-epoch 1\
    --batch_size 8 \
    --device '0'\
    --backbone 'PVT'\
    --add_module 'SoftHGNN-SeS'    
```
## ObjectDetection
