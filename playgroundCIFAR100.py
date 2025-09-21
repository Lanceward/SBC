import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model import spiking_resnet, train_classify, spiking_vgg
import argparse

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder, CIFAR100
from archs.cifarsvhn.resnet import ResNet19
from archs.cifarsvhn.vgg import vgg16_bn
from data_utils import Cutout

def build_model(device, detach, class_num: int = 1000):
    """Return a spiking VGG-16 BN with ImageNet weights already loaded."""
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,                       # grabs torchvision weights
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=detach,
        v_threshold=1.0,
        num_classes=class_num
    ).to(device).eval()
    with torch.no_grad():
        net.fc.bias.zero_()
    functional.set_step_mode(net, 'm')         # multi-step forward
        
    return net

class CIFAR100Trainer(train_classify.Trainer):
    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def set_optimizer(self, args, parameters):
        opt_name = args.opt.lower()
        if opt_name.startswith("adamax"):
            optimizer = torch.optim.Adamax(parameters, lr=args.lr, weight_decay=args.weight_decay)
            return optimizer
        else:
            return super().set_optimizer(args, parameters)

    def set_lr_scheduler(self, args, optimizer):
        lr_scheduler = args.lr_scheduler.lower()
        if lr_scheduler =="multistep":
            print("multistep scheduler: " + str(args.lr_milestones), args.lr_gamma)
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
            return main_lr_scheduler
        else:
            return super().set_lr_scheduler(args, optimizer)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--T', type=int, help="total time-steps")
        parser.add_argument('--detach', action='store_true', help="detach")
        parser.add_argument("--lr-milestones", nargs="+", default=[150, 255], type=int, help="Epochs at which to decay the learning rate")
        return parser

    def load_data(self, args):
        return self.loadCIFAR100(args)

    def loadCIFAR100(self, args):
        """Return DataLoader for Cifar100 val/ subset."""
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2673, 0.2564, 0.2762]
        tf_train_cifar100 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        dataset_train = CIFAR100(root=args.data_path, transform=tf_train_cifar100, train=True, download=True)
        dataset_test = CIFAR100(root=args.data_path, transform=valid_transform, train=False, download=True)
        
        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            loader_g = torch.Generator()
            loader_g.manual_seed(args.seed)
            train_sampler = torch.utils.data.RandomSampler(dataset_train, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
        return dataset_train, dataset_test, train_sampler, test_sampler

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

class CIFAR100TrainerResNet19(CIFAR100Trainer):
    def load_model(self, args, num_classes):
        """Return a spiking ResNet19 BN with ImageNet weights already loaded."""
        net = ResNet19(
            num_classes=num_classes
        ).to(args.device).eval()
        functional.set_step_mode(net, 'm')         # multi-step forward
        return net

class CIFAR100TrainerVGG16(CIFAR100Trainer):
    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--tau', default=4.0/3.0, type=float, help="LIF time constant")
        return parser
    
    def load_model(self, args, num_classes):
        """Return a spiking VGG-16 BN with ImageNet weights already loaded."""
        net = vgg16_bn(
            num_classes=num_classes
        ).to(args.device).eval()
        functional.set_step_mode(net, 'm')         # multi-step forward
        return net

def main():
    trainer = CIFAR100TrainerResNet19()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)


if __name__ == "__main__":
    main()