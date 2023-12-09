# SFA3D_demo : Application of SFA3D-based specific road driving data

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

- This code is based on [SFA3D](https://github.com/maudzung/SFA3D.git)

## Features
- Application of special road driving data based on Super fast and accurate 3D object detection based on LiDAR 
- [Here](https://github.com/maudzung/SFA3D.git)

<img width="1067" alt="image" src="https://github.com/EASYhz/SFA3D_demo/assets/65584699/7397f415-ce9a-4ad1-9188-9da3fed3ad2c">


## Initialization
 - clone the repository
``` bash
$ git clone https://github.com/EASYhz/SFA3D_demo.git
```

## Getting Started
### 1. Requirement
- To run, download the required package through the following command. (Recommend virtual environment)

```  bash
$ pip install -r requirements.txt
```

### 2. Dataset
- Download the Special road driving dataset from the AI HuB. [here](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71258)

<img width="1267" alt="image" src="https://github.com/EASYhz/SFA3D_demo/assets/65584699/04694181-acf8-4e03-949e-461abaebdd45">


### 3. How To Run

#### 3.1 Dataset file structure
- Create dataset file structure
```bash
$ python sfa/data_process/demo_dataset.py
```

##### 3.1.2 Making demonstration
``` bash
$ cd sfa
$ python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.1
```

#### 3.2 `.pcd` To `.bin`
- ***The process of putting the downloaded dataset is required.***
- The [SFA3D](https://github.com/maudzung/SFA3D.git) model we referenced is based on the bin file. Currently, the downloaded dataset is pcd. The process of converting it is necessary.
- `pcd_path`, `bin_path`, `file_name` have default values

```bash
$ python pcd2bin.py --pcd_path={path of input pcd file directory} --bin_path={path of output bin file directory}  --file_name={name of bin file}
```

#### 3.3 Making demonstration
``` bash
$ cd sfa
$ python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.1
```


#### 3.4. Training
- More information below can be found [here](https://github.com/maudzung/SFA3D.git).

##### 3.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0
```

##### 3.4.2. Distributed Data Parallel Training
- **Single machine (node), multiple GPUs**

```
python train.py --multiprocessing-distributed --world-size 1 --rank 0 --batch_size 64 --num_workers 8
```

- **Two machines (two nodes), multiple GPUs**

   - _**First machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 0 --batch_size 64 --num_workers 8
    ```

   - _**Second machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 1 --batch_size 64 --num_workers 8
    ```

#### Tensorboard

- To track the training progress, go to the `logs/` folder and 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

## References

[1] SFA3D: [SFA3D](https://github.com/maudzung/SFA3D.git) <br>
[2] pcd2bin: [pcd2bin with python2.7](https://github.com/Yuseung-Na/pcd2bin) <br>
[3] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[4] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[5] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[6] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[5] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)


## Folder structure

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   ├── lidar2/
        │   ├── bin2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── lidar2/
        │   ├── bin2/
        │   └── velodyne/
        │
        └── classes_names.txt
└── sfa/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   ├── kitti_data_utils.py
    │   └── pcd2bin.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```


[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
