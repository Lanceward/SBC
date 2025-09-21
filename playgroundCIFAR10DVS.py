import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model import spiking_resnet, train_classify, spiking_vgg
from spikingjelly.datasets import cifar10_dvs, split_to_train_test_set

import argparse

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder, CIFAR100
from archs.cifarsvhn.resnet import ResNet19
from archs.cifarsvhn.vgg import vgg16_bn
from data_utils import Cutout

from modelutils import *
from model_library import DVSCIFAR10NET_DOWNSIZED, DVSCIFAR10NET_FULLSIZED

class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)

class CIFAR10DVSTrainer(train_classify.Trainer):
    # def preprocess_train_sample(self, args, x: torch.Tensor):
    #     # define how to process train sample before send it to model
    #     return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    # def preprocess_test_sample(self, args, x: torch.Tensor):
    #     # define how to process test sample before send it to model
    #     return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def set_optimizer(self, args, parameters):
        opt_name = args.opt.lower()
        if opt_name == "adam":
            print("Adam optimizer chosen", args.lr)
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
            return optimizer
        else:
            return super().set_optimizer(args, parameters)

    def set_lr_scheduler(self, args, optimizer):
        lr_scheduler = args.lr_scheduler.lower()
        if lr_scheduler == "multistep":
            print("multistep scheduler: " + str(args.lr_milestones), args.lr_gamma)
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
            return main_lr_scheduler
        else:
            return super().set_lr_scheduler(args, optimizer)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--T', type=int, help="total time-steps")
        parser.add_argument('--tau', default=2.0, type=float, help="LIF time constant")
        parser.add_argument('--detach', action='store_true', help="detach")
        parser.add_argument("--lr-milestones", nargs="+", default=[150, 255], type=int, help="Epochs at which to decay the learning rate")
        return parser

    def load_model(self, args, num_classes):
        """Return a spiking VGG-16 BN with ImageNet weights already loaded."""
        net = DVSCIFAR10NET_FULLSIZED(
            channels=128,
            tau=args.tau
        ).to(args.device).eval()
        functional.set_step_mode(net, 'm')         # multi-step forward
        return net

    def load_data(self, args):
        return self.loadCIFAR10DVS(args)

    def loadCIFAR10DVS(self, args):
        print("loadCIFAR10DVS")
        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        full_dataset = cifar10_dvs.CIFAR10DVS(
            root=args.data_path,
            data_type="frame",
            frames_number=args.T,
            split_by="number",
            transform=None,
            duration=10000,
        )
        print("split")

        dataset_train, dataset_test = split_to_train_test_set(
            train_ratio=0.9,
            origin_dataset=full_dataset,
            num_classes=10,
            random_split=False,
        )
        
        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            loader_g = torch.Generator()
            loader_g.manual_seed(args.seed)
            train_sampler = torch.utils.data.RandomSampler(dataset_train, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
        dataset_train.classes = full_dataset.classes
        
        return dataset_train, dataset_test, train_sampler, test_sampler

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

def main():
    trainer = CIFAR10DVSTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)


if __name__ == "__main__":
    main()