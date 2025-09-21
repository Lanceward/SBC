import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model import spiking_resnet, train_classify, spiking_vgg, sew_resnet
import argparse

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from data_utils import Cutout
from tqdm import tqdm
from collections import OrderedDict
import re

class ClassificationPresetEval:
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


class ImageNetTrainer(train_classify.Trainer):
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

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

class ImageNetTrainerResNet18(ImageNetTrainer):
    def load_model(self, args, num_classes):
        """Return a spiking ResNet19 BN with ImageNet weights already loaded."""
        net = sew_resnet.sew_resnet18(
            pretrained=True,
            cnf='ADD',                                 # 'ADD', 'AND', or 'IAND'
            spiking_neuron=neuron.LIFNode,     # Integrate‑and‑fire
            v_threshold=1.0,
            surrogate_function=surrogate.ATan(),       # surrogate gradient
            detach_reset=True,
        ).to(args.device).eval()
        functional.set_step_mode(net, 'm')         # multi-step forward
        return net

def RemapSEWResNetCheckpoints(checkpoint):
    sd_old = checkpoint.get("state_dict", checkpoint)

    sd_old = {k.replace('module.', ''): v for k, v in sd_old.items()}

    # 2) remap all the Sequential(conv, bn) keys into convX / bnX
    sd_mapped = {}
    for k, v in sd_old.items():
        new_k = None

        # initial conv1 / bn1
        m = re.match(r"^conv1\.0\.(weight|bias)$", k)
        if m:
            new_k = f"conv1.{m.group(1)}"
        m = re.match(r"^conv1\.1\.(weight|bias|running_mean|running_var|num_batches_tracked)$", k)
        if m:
            new_k = f"bn1.{m.group(1)}"

        # blocks: layer{n}.{i}.conv{1|2} / bn{1|2}
        m = re.match(r"^(layer\d+\.\d+)\.conv([123])\.0\.(weight|bias)$", k)
        if m:
            layer, num, attr = m.groups()
            new_k = f"{layer}.conv{num}.{attr}"
        m = re.match(
            r"^(layer\d+\.\d+)\.conv([123])\.1\.(weight|bias|running_mean|running_var|num_batches_tracked)$",
            k,
        )
        if m:
            layer, num, attr = m.groups()
            new_k = f"{layer}.bn{num}.{attr}"

        # downsample in blocks
        m = re.match(r"^(layer\d+\.\d+)\.downsample\.0\.0\.(weight|bias)$", k)
        if m:
            layer, attr = m.groups()
            new_k = f"{layer}.downsample.0.{attr}"
        m = re.match(
            r"^(layer\d+\.\d+)\.downsample\.0\.1\.(weight|bias|running_mean|running_var|num_batches_tracked)$",
            k,
        )
        if m:
            layer, attr = m.groups()
            new_k = f"{layer}.downsample.1.{attr}"

        # otherwise leave it (e.g. fc.weight, fc.bias)
        if new_k is None:
            new_k = k

        sd_mapped[new_k] = v
    return sd_mapped

def validateResnet18():
    T = 4
    device = 'cuda'
    val_crop_size = 224
    batch_size_ = 32
    model_path = "models/imagenet/sew18_checkpoint_319.pth"
    
    """Return a spiking ResNet19 BN with ImageNet weights already loaded."""
    net = sew_resnet.sew_resnet18(
        pretrained=False,
        cnf='ADD',                                 # 'ADD', 'AND', or 'IAND'
        spiking_neuron=neuron.IFNode,     # Integrate‑and‑fire
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(),       # surrogate gradient
        detach_reset=True,
    ).to(device).eval()

    print(net)

    # # 1. Load the raw checkpoint
    ckpt = torch.load(model_path, map_location=device,  weights_only=False)
    ckpt = ckpt['model']

    sd_mapped = RemapSEWResNetCheckpoints(ckpt)

    # 4. Load weights (conv/bn layers will match exactly)
    missing, unexpected = net.load_state_dict(sd_mapped, strict=False)
    print('Missing keys (will be neuron-only):', missing)
    print('Unexpected keys:', unexpected)
    functional.set_step_mode(net, 'm')         # multi-step forward
    
    # load validation dataset
    valdir = "../datasets/ImageNet/val"
    dataset_test = ImageFolder(
        valdir,
        ClassificationPresetEval(crop_size=val_crop_size),
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size_, sampler=test_sampler, num_workers=8, pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_correct = 0.0
    val_samples = 0
    # validate:
    with torch.no_grad():
        for image, target in tqdm(data_loader_test):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_t = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            output_t = net(image_t)
            output = output_t.mean(0)
            # loss = criterion(output, target)
            val_correct += (output.argmax(dim=1) == target).float().sum().item()
            val_samples += target.size(0)
            functional.reset_net(net)

    print(val_correct/val_samples)
        
def main():
    trainer = ImageNetTrainerResNet18()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)


if __name__ == "__main__":
    # main()
    validateResnet18()