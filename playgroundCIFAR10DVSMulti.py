#!/usr/bin/env python3
"""
Distributed training script for CIFAR10-DVS on two GPUs (cuda:1 and cuda:2).

Launch example:
    CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone \
        --nproc_per_node=2 train_ddp.py \
        -T 100 -b 64 --amp

`torchrun` automatically sets the LOCAL_RANK / RANK / WORLD_SIZE environment
variables. With the mask `CUDA_VISIBLE_DEVICES=1,2`, rank 0 maps to GPU1 and
rank 1 maps to GPU2.
"""

import os
import time
import argparse
import sys
import datetime
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch import amp
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

import numpy as np

from spikingjelly.activation_based import functional
from spikingjelly.datasets import cifar10_dvs, split_to_train_test_set

from modelutils import *  # noqa: F403,F401 – supplies e.g. TORCH_SEED
from model_library import DVSCIFAR10NET_FULLSIZED

# if 'cuda' in DEV:
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False

# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def spike_rate_loss(true_rate: float, false_rate: float, model_out: torch.Tensor, label: torch.Tensor) -> torch.Tensor:  # noqa: E501
    """MSE loss between model spike‑rate output and class‑dependent target rates."""
    return F.mse_loss(model_out, label)


def init_distributed(local_rank: int, backend: str = "nccl") -> None:
    """Initialise torch.distributed and set the current CUDA device."""
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR10‑DVS Distributed Training")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)),
                        help="Local rank supplied by torchrun")
    parser.add_argument("-T", type=int, default=100, help="Simulation time‑steps")
    parser.add_argument("-b", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("-epochs", type=int, default=100, metavar="N", help="Total epochs")
    parser.add_argument("-j", type=int, default=4, metavar="N", help="Data‑loading workers")
    parser.add_argument("-data-dir", type=str, default="./datas/CIFAR10DVS", help="Dataset root")
    parser.add_argument("-out-dir", type=str, default="./logs", help="Logs & checkpoints root")
    parser.add_argument("-resume", type=str, help="Path to checkpoint to resume")
    parser.add_argument("-amp", action="store_true", help="Enable mixed‑precision (AMP)")
    parser.add_argument("-opt", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer")
    parser.add_argument("-momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-wd", type=float, default=1e-4, help="Momentum for SGD")    
    parser.add_argument("-tau", type=float, default=2.0, help="Tau for LIF neurons")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed initialisation
    # ------------------------------------------------------------------
    init_distributed(args.local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", args.local_rank)

    if rank == 0:
        print("Args:", args)
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {args.local_rank}")

    # ------------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------------
    net = DVSCIFAR10NET_FULLSIZED(channels=128, tau=args.tau, detach_reset=True).to(device)
    functional.set_step_mode(net, "s")
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    scaler: Optional[amp.GradScaler] = amp.GradScaler(enabled=args.amp)

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * 0.40, args.epochs * 0.60, args.epochs * 0.75])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    full_dataset = cifar10_dvs.CIFAR10DVS(
        root=args.data_dir,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
        transform=None,
    )

    # Reproducibility
    try:
        np.random.seed(TORCH_SEED)  # Provided by modelutils
    except NameError:
        np.random.seed(42)

    train_dataset, test_dataset = split_to_train_test_set(
        train_ratio=0.9,
        origin_dataset=full_dataset,
        num_classes=10,
        random_split=False,
    )

    train_sampler = data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.b,
        sampler=train_sampler,
        num_workers=args.j,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.b,
        sampler=test_sampler,
        num_workers=args.j,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Checkpoint & logging
    # ------------------------------------------------------------------
    start_epoch = 0
    max_test_acc = -1.0

    out_dir = os.path.join(args.out_dir,
                           f"{type(net.module).__name__}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_ddp")
    if args.amp:
        out_dir += "_amp"
    out_dir += "_dvs"

    if rank == 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Rank 0] Created log dir: {out_dir}")

    if args.resume is not None:
        map_loc = {"cuda:%d" % 0: f"cuda:{args.local_rank}"}
        checkpoint = torch.load(args.resume, map_location=map_loc)
        net.module.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc = checkpoint.get("max_test_acc", max_test_acc)
        if rank == 0:
            print(f"[Rank 0] Resumed from {args.resume} @ epoch {start_epoch}")

    writer = SummaryWriter(out_dir, purge_step=start_epoch) if rank == 0 else None

    # ----------------------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------------------
    #autocast_args = dict(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp)
    #print("amp: ", autocast_args)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        net.train()

        ep_loss = 0.0
        ep_correct = 0.0
        ep_samples = 0

        idx = 0
        for frame, label in train_loader:
            print("\r" + str(idx) + "/" + str(len(train_loader)), end='')
            idx+=1
            frame = frame.to(device, non_blocking=True).transpose(0, 1)  # T,N,C,H,W
            label = label.to(device, non_blocking=True)
            label_onehot = F.one_hot(label, 10).float()

            optimizer.zero_grad(set_to_none=True)

            #with amp.autocast(**autocast_args):
            out_fr = 0.0
            for t in range(args.T):
                out_fr += net(frame[t])
            out_fr /= args.T
            loss = spike_rate_loss(0.82, 0.02, out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                ep_samples += label.size(0)
                ep_loss += loss.item() * label.size(0)
                ep_correct += (out_fr.argmax(dim=1) == label).float().sum().item()
                functional.reset_net(net)

        scheduler.step()

        # Reduce metrics across processes
        metrics = torch.tensor([ep_loss, ep_correct, ep_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        train_loss = metrics[0].item() / metrics[2].item()
        train_acc = metrics[1].item() / metrics[2].item()

        if rank == 0 and writer:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)

        # ----------------------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------------------
        net.eval()
        val_loss = 0.0
        val_correct = 0.0
        val_samples = 0

        with torch.no_grad():#, amp.autocast(**autocast_args):
            for frame, label in test_loader:
                frame = frame.to(device, non_blocking=True).transpose(0, 1)
                label = label.to(device, non_blocking=True)
                label_onehot = F.one_hot(label, 10).float()

                out_fr = 0.0
                for t in range(args.T):
                    out_fr += net(frame[t])
                out_fr /= args.T

                loss = spike_rate_loss(0.82, 0.02, out_fr, label_onehot)

                val_samples += label.size(0)
                val_loss += loss.item() * label.size(0)
                val_correct += (out_fr.argmax(dim=1) == label).float().sum().item()
                functional.reset_net(net)

        # Aggregate validation metrics
        val_metrics = torch.tensor([val_loss, val_correct, val_samples], device=device)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        test_loss = val_metrics[0].item() / val_metrics[2].item()
        test_acc = val_metrics[1].item() / val_metrics[2].item()

        if rank == 0 and writer:
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_acc", test_acc, epoch)

        # ----------------------------------------------------------------------------------
        # Checkpointing (rank 0 only)
        # ----------------------------------------------------------------------------------
        if rank == 0:
            save_dict = {
                "net": net.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "max_test_acc": max(max_test_acc, test_acc),
            }
            torch.save(save_dict, os.path.join(out_dir, "checkpoint_latest.pth"))
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(save_dict, os.path.join(out_dir, "checkpoint_max.pth"))

            print()
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} max_acc={max_test_acc:.4f}"
                f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

    if rank == 0 and writer:
        writer.close()


if __name__ == "__main__":
    main()
