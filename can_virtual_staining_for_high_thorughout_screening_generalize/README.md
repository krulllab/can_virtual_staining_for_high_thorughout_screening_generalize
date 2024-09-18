# Can virtual staining for High-throughput screening generalize?
<img src='imgs/dataset.png' align="center" width=1000>


## [Paper](To be added) [Dataset](To be added) <br>

## Pytorch implementation of adapted pix2pixHD method for high-resolution (e.g. 1080x1080) virtual staining via image-to-image translation.
## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started

Create a new environment
```bash
conda create -n can_virtual_staining_for_high_thorughout_screening_generalize python=3.8
conda activate can_virtual_staining_for_high_thorughout_screening_generalize
pip install -e .
```

Clone this repo:
```bash
git clone git@github.com:krulllab/can_virtual_staining_for_high_thorughout_screening_generalize.git
cd src
```

### Training
```bash
python ./src/train.py --dataroot ../path_to_data/ --data_type 16 --batchSize 4 --checkpoints_dir ../results/ --label_nc 0 --name experiment1 --no_instance  --resize_or_crop none --input_nc 1 --output_nc 1 --seed 42 --no_vgg_loss  --nThreads 1 --loadSize 256 --ndf 32 --norm instance --use_dropout  --fp16 --gpu_ids 1
```
- To view training results, please launch `tensorboard --logdir opt.checpoints_dir`

### Multi-GPU training
```bash
python ./src/train.py --dataroot ../path_to_data/ --data_type 16 --batchSize 4 --checkpoints_dir ../../results --label_nc 0 --name experiment1 --no_instance  --resize_or_crop none --input_nc 1 --output_nc 1 --seed 42 --no_vgg_loss  --nThreads 1 --loadSize 256 --ndf 32 --norm instance --use_dropout  --fp16 --gpu_ids 1,2,3
```
### Training with Automatic Mixed Precision (AMP) for faster speed
- To train with mixed precision support, please first install apex from: https://github.com/NVIDIA/apex
- You can then train the model by adding `--fp16`. For example,
```bash
python ./src/train.py --dataroot ../path_to_data/ --data_type 16 --batchSize 4 --checkpoints_dir ../results/experiment1/ --fp16
```
### Testing
```bash

python test.py --results_dir ../results/inference/ --dataroot ../path_to_data/ --data_type 16 --batchSize 1 --checkpoints_dir ../results/experiment1/
```
The test results will be saved to a html file here: `./results/


## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
