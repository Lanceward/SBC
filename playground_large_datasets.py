import torch
from spikingjelly.activation_based.model import spiking_vgg, spiking_resnet
from spikingjelly.activation_based import functional, neuron, surrogate, ann2snn, monitor
import os
import torchvision
import torchvision.transforms as transforms

from osbc_prune import *
import config_sq
from data_utils import data_transforms
from modelutils import *

from torchvision.datasets import ImageFolder, CIFAR100
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

MODEL_SAVE_PATH = "./models/cifar100"
BATCH=32

def build_model(device, neuron_name: str, detach: bool = True):
    """Return a spiking VGG-16 BN with ImageNet weights already loaded."""
    Node = getattr(neuron, neuron_name)        # IFNode or LIFNode
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,                       # grabs torchvision weights
        spiking_neuron=Node,
        surrogate_function=surrogate.ATan(),
        detach_reset=detach,
        v_threshold=1.0,
        num_classes=1000
    ).to(device).eval()
    with torch.no_grad():
        net.fc.bias.zero_()
    functional.set_step_mode(net, 'm')         # multi-step forward
        
    return net

def build_ann_model(device, pretrained: bool = False):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    return net

def cifar100_loader(root, train, batch_size, workers):
    """Return DataLoader for ImageNet val/ subset."""
    tf_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    #ds = ImageFolder(os.path.join(root, 'val'), transform=tf_val)
    ds = CIFAR100(root=root, transform=tf_val, train=train)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=True, num_workers=workers,
                        pin_memory=True)
    return loader

def imagenet_loader(root, train, batch_size, workers):
    """Return DataLoader for ImageNet val/ subset."""
    tf_val = transforms.Compose([
        transforms.Resize(232, interpolation=InterpolationMode.BILINEAR), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.PILToTensor(),
        #transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    if train:
        ds = ImageFolder(os.path.join(root, 'train'), transform=tf_val)
    else:
        ds = ImageFolder(os.path.join(root, 'val'), transform=tf_val)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=workers,
                        pin_memory=False)
    return loader

@torch.no_grad()
def validate(net, loader, device, T_steps: int, val_batch_size: int, fr_monitor):
    """Return Top-1 accuracy (%) of *net* on *loader* at simulation length T."""
    correct = seen = 0
    val_num=0
    for imgs, labels in tqdm(loader, unit='batch', desc='validate'):
        if val_num == val_batch_size:
           break
        val_num+=1
                
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
                
        x_seq = imgs.unsqueeze(0).repeat(T_steps, 1, 1, 1, 1)  # [T,N,C,H,W]

        out_seq = net(x_seq)               # [T,N,1000]
        mean_spike_per_neuron = out_seq.sum(0).mean(0)
        # Histogram over spike count values (you want 20 bins)
        num_bins = 20
        hist = torch.histc(mean_spike_per_neuron, bins=num_bins, min=0.0, max=mean_spike_per_neuron.max().item())
                           
        logits = out_seq.mean(0)           # rate decoding
        pred = logits.argmax(1)
        print(torch.mean(imgs), torch.var(imgs), fr_monitor[0], hist.tolist())   # [N] tensor
        #print(pred, labels)
        fr_monitor.clear_recorded_data()

        correct += (pred == labels).sum().item()
        seen    += labels.size(0)

        functional.reset_net(net)          # clear membrane state

    return 100. * correct / seen

@torch.no_grad()
def validate_ann(net, loader, device, val_batch_size: int):
    """Return Top-1 accuracy (%) of *net* on *loader*"""
    correct = seen = 0
    for imgs, labels in tqdm(loader, unit='batch', desc='validate'):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        
        out_seq = net(imgs)               # [N,1000]
                           
        pred = out_seq.argmax(1)
        #print(torch.mean(imgs), torch.var(imgs))   # [N] tensor
        #print(pred, labels)

        correct += (pred == labels).sum().item()
        seen    += labels.size(0)

    return 100. * correct / seen

def first_lif_tau(net):
    for m in net.modules():
        if isinstance(m, neuron.IFNode):
            return m.tau if torch.is_tensor(m.tau) else torch.tensor(m.tau)
    raise RuntimeError("No LIFNode found!")

def main():
    p = argparse.ArgumentParser('Spiking VGG-16 ImageNet validation')
    p.add_argument('--data', required=True, help='ImageNet root dir (with train/ val/)')
    p.add_argument('-b', '--batch-size', default=64, type=int)
    p.add_argument('-t', '--timesteps', default=8, type=int, help='simulation length T')
    p.add_argument('-j', '--workers', default=4, type=int)
    p.add_argument('--neuron', default='IFNode', choices=['IFNode', 'LIFNode'])
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--convert', action='store_true')
    args = p.parse_args()

    device = torch.device('mps')

    if args.convert:
        print(114)
        ann_net = build_ann_model(device, True)
        
        val_loader = imagenet_loader(args.data, False, args.batch_size, args.workers)

        # acc1 = validate_ann(ann_net, val_loader, device, 100)
        # print(f'ANN Top-1 accuracy @T={args.timesteps}: {acc1:.2f} %')

        # convert
        train_loader = imagenet_loader(args.data, True, args.batch_size, args.workers)
        model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
        snn_net = model_converter(ann_net)
        functional.set_step_mode(snn_net, 'm')         # multi-step forward

        # --- save ---
        torch.save(snn_net, "snn_resnet18.pt")          # gm is your spiking-ResNet GraphModule
    else:
        print(161)
        # --- load ---
        snn_net = torch.load("snn_resnet18.pt", map_location="cpu", weights_only=False).to(device)

        # 2. Build an empty single-step SpikingResNet-18
        sj = spiking_resnet.spiking_resnet18(
                spiking_neuron=neuron.LIFNode,
                v_threshold=1.,              # will be overwritten
                tau=2.0,
                surrogate_function=surrogate.ATan())

        # 3. Copy parameters **by traversal order**
        with torch.no_grad():
            for m_fx, m_sj in zip(snn_net.modules(), sj.modules()):
                # conv / bn / fc
                if isinstance(m_fx, (torch.nn.Conv2d,
                                    torch.nn.BatchNorm2d,
                                    torch.nn.Linear)):
                    m_sj.load_state_dict(m_fx.state_dict(), strict=False)
                # LIF → IF name mapping: copy threshold & decay
                if isinstance(m_fx, neuron.LIFNode):
                    m_sj.v_threshold.data.copy_(m_fx.v_threshold.data)
                    m_sj.tau = m_fx.tau
        
        val_loader = imagenet_loader(args.data, False, args.batch_size, args.workers)
    
        def firing_rate(s_seq: torch.Tensor):
            # s_seq.shape = [T, N, …]; average over time and neurons in batch
            return s_seq.flatten(1).mean(1)            # → [N]   (one value per sample)
        fr_monitor        = monitor.OutputMonitor(sj, neuron.IFNode,            # rate (%)
                                                function_on_output=firing_rate)   # :contentReference[oaicite:0]{index=0}
        

        acc1 = validate(sj, val_loader, device, T_steps=8, val_batch_size=100, fr_monitor=fr_monitor)
        print(f'ANN Top-1 accuracy @T={args.timesteps}: {acc1:.2f} %')

if __name__ == '__main__':
    main()