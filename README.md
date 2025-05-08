# SoftHGNN
Implementation of the paper "SoftHGNN: Soft Hypergraph Neural Networks for General Visual Recognition".
# Getting Started🚀
## 1.Classification
## 2.CrowdCouting
### Data Preparation
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
### Training
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
## 3.ObjectDetection
```
from ultralytics import YOLO

model = YOLO('/yolov12_softhgnn/yolov12n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()
```

