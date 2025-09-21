# snn_quant

Post Training Quantization/Pruning with Second-order Hessian Information for Spiking Neural Network

W

Required packages: 
```bash
pip install spikingjelly torchvision numpy
```

Create folders:
```bash
mkdir datas/NMNIST/download datas/CIFAR10DVS/download datas/DVS128Gesture/download 
```
follow instructions to manually put NMNIST dataset in datas/NMNIST/download folder, and dvs128 gesture dataset in datas/DVS128Gesture/download folder

TRAINING

Training commands: 

N-MNIST:
```bash
python3 playgroundNMNIST.py -T 100 -device mps -b 64 -epochs 200 -data-dir ./datas -opt adam -lr 1e-3 -tau 2.0
```
DVS128-Gesture:
```bash
python3 playgroundDVS128.py -T 20 -device mps -b 16 -epochs 512 -opt adam -lr 1e-3 -amp -tau 2.0
```

CIFAR10-DVS:
```bash
python3 playgroundCIFAR10DVS.py -T 20 -device mps -b 16 -epochs 512 -opt adam -lr 1e-3 -amp -tau 2.0
```

for cuda backend, replace ```-device mps``` with ```-device cuda```, and uncomment modelutils.py line 5, comment line 6


PRUNING

Pruning is configured in the compression_XXX.py files. At the top of each file, there is MODEL_SAVE_PATH, which is where you save the files to. TARGET_PRUNE is the final targeted sparsity. OSBS=True means the algorithm is in OSBC pruning mode; OSBS=False means the algorithm is in OBC pruning mode.

In main, there is model_path and data_path, which tells you where the model and datasets are stored. In osbc_prune.py, MBP controls whether the algorithm is in magnitude based pruning mode, which overrides OSBS if MPS=True. 

For ImageNet, download SEW-RESNET checkpoints from https://figshare.com/articles/software/Spike-Element-Wise-ResNet/14752998. Download the sew18, 50, 152 checkpoint.

Pruning Commands:

N-MNIST:
```bash
python3 compression_nmnist.py
```

DVS128-Gesture:
```bash
python3 compression_dvs128.py
```

CIFAR10-DVS:
```bash
python3 compression_cifar10dvs.py
```

CIFAR100:
```bash
python3 compression_cifar100.py
```

IMAGENET:
```bash
python3 compression_imagenet.py
```

QUANTIZATION

Quantization is very similar to pruning, with TARGET_BIT_WIDTH controlling the quantization bit width. In osbc_quant.py, RTN=True overwrites OSBS setting, and the algorithm rounds to nearest quantization grid

Quantization Commands:

N-MNIST:
```bash
python3 quantization_nmnist.py
```
DVS128-Gesture:
```bash
python3 quantization_dvs128.py
```
CIFAR10-DVS:
```bash
python3 quantization_cifar10dvs.py
```

Training Resnet19 with CIFAR100
```
python3 playgroundCIFAR100.py --T 5 --model spiking_resnet19 --data-path ./datas --batch-size 128 --lr 0.1 --epochs 300 --pretrained --device mps --lr-scheduler multistep --lr-milestones 150 225 --wd 1e-4
```

