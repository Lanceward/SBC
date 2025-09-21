import os, argparse, random
from playgroundImageNet import *
from osbc_prune import *
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder, CIFAR100
import torchvision.transforms as transforms
from compression_imagenet import RepeatTransform, FlattenedSEWResNet
# parallellism
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

BATCH=32

TARGET_PRUNE = "0.647"

MODEL_SAVE_PATH = "./models/imagenet"

def loadImageNet(data_path, T):
    """Return DataLoader for Cifar100 val/ subset."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    crop_size = 224
    resize_size = 256
    interpolation = InterpolationMode.BILINEAR
    tf_train_imagenet = transforms.Compose([
        # transforms.Resize(resize_size, interpolation=interpolation),
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=MEAN, std=STD),
        RepeatTransform(T),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=MEAN, std=STD),
        RepeatTransform(T),
    ])
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")
    dataset_train = ImageFolder(root=traindir, transform=tf_train_imagenet)
    dataset_test = ImageFolder(root=valdir, transform=valid_transform)
    
    return dataset_train, dataset_test
# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_batchnorm(model: nn.Module, *, affine: bool = True) -> None:
    """
    Puts every BatchNorm* layer in eval mode (no running-stat updates).
    Optionally freezes the affine weights/biases as well.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()                # 1️⃣ stop running-stat updates
            if affine:              # 2️⃣ optionally freeze γ, β
                m.weight.requires_grad = False
                m.bias.requires_grad  = False

def apply_mask_hook(param: torch.Tensor, mask: torch.Tensor):
    """Block gradients for pruned weights."""
    assert mask.shape == param.shape
    def hook(grad):
        return grad * mask.to(param.device)
    param.register_hook(hook)


def inspect_bns(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            print(f"— {name} —")
            print(f"  train mode     : {m.training}")
            print(f"  weight grad    : {m.weight.requires_grad}")
            print(f"  bias   grad    : {m.bias  .requires_grad}")
            # show a summary statistic of the running stats
            print(f"  running_mean   : {m.running_mean.mean().item():.4f} ± {m.running_mean.std().item():.4f}")
            print(f"  running_var    : {m.running_var.mean().item():.4f} ± {m.running_var.std().item():.4f}")
            print(f"  momentum       : {m.momentum}\n")

def inspect_layer_sparsity(flat_model):
    masks = {}
    total_zero_count = 0
    total_weight_count = 0
    for name, layer in find_layers(flat_model.layer).items():
        w = layer.weight
        masks[name] = torch.abs(w) > 5e-7
        layer_weight_nonzero = torch.masked_select(w, masks[name])
        total_zero_count += w.numel()-len(layer_weight_nonzero)
        total_weight_count += w.numel()            
        print(f'Layer {name}, {1.0-(torch.sum(masks[name]) / w.numel()).item():.5f} {torch.mean(layer_weight_nonzero).item():.5f} {torch.var(layer_weight_nonzero).item():.5f}')
    print(f'Total sparsity: {total_zero_count}, {total_weight_count}, {total_zero_count/total_weight_count:.4f}')

# --------------------------------------------------------------------------- #
#  worker - one per GPU
# --------------------------------------------------------------------------- #
def main(local_rank: int, args):
    # ---------- distributed initialisation ----------
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_random_seed(TORCH_SEED + local_rank)

    # ---------- build / load model ----------
    model_path = f"./models/imagenet/{TARGET_PRUNE}_pruned_model.pth"
    model = sew_resnet.sew_resnet18(
        pretrained=False,
        cnf='ADD',
        spiking_neuron=neuron.IFNode,
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=False,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.state_dict(), strict=False)
    if local_rank == 0:
        print('Missing keys (neuron only):', missing)
        print('Unexpected keys:', unexpected)
    freeze_batchnorm(model, affine=False)

    # flatten helper _before_ wrapping in DDP so attributes still resolve
    flat = FlattenedSEWResNet(model).eval()
    # Compute masks once (they’re identical on every rank)
    model_masks = {}
    for name, layer in find_layers(flat.layer).items():
        w = layer.weight
        model_masks[name] = torch.abs(w) > 5e-7
    if local_rank == 0:
        inspect_layer_sparsity(flat)
    
    # ---------- DistributedDataParallel ----------
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    flat  = FlattenedSEWResNet(model.module).eval()   # point at wrapped model

    # attach gradient mask hooks (must run *after* DDP wrapping)
    for name, mask in model_masks.items():
        apply_mask_hook(flat.layer[name].weight, mask)

    # ---------- data ----------
    T = 4
    lr_ = 1e-3
    batch_size_ = 32
    wd_ = 0.0#1e-5
    epoch_ = 5
    data_path = "../datasets/ImageNet"
    train_ds, test_ds = loadImageNet(data_path, T)
    # train_ds, _ = stratified_split(train_ds, frac=0.2)

    # Each GPU sees a unique shard
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    test_sampler  = DistributedSampler(test_ds, shuffle=False)

    finetune_loader = DataLoader(
        train_ds, batch_size=batch_size_, sampler=train_sampler,
        num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )

    if local_rank == 0:
        model.eval()
        pre_finetune = 0.61356
        # pre_finetune = validate_model(
        #     model.module,                    # unwrap DDP
        #     T, test_ds, pruning_perc=f"{TARGET_PRUNE}_tuned",
        #     out_dir="", batch=BATCH,
        # )
        print("pre_finetune", pre_finetune)

    # ---------- optimiser / loss ----------
    optimizer  = torch.optim.SGD(model.parameters(), lr=lr_,
                                 momentum=0.9, weight_decay=wd_)
    # scheduler -------------------------------------------
    total_steps  = epoch_ * len(finetune_loader)   # per-iteration stepping
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_,        # peak LR
        total_steps=total_steps,
        pct_start=0.15,      # 15 % warm-up (linear rise)
        anneal_strategy="cos",
        div_factor=10,      # start LR = max_lr / 10
        final_div_factor=100,  # final LR = max_lr / 100
    )
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.0)
    functional.set_step_mode(model, 'm')
    model.train()
    freeze_batchnorm(model,affine=False)
    
    for e in range(epoch_):
        # ---------- one epoch fine-tune ----------
        train_sampler.set_epoch(e)          # shuffle shards the same way on every rank
        pbar = tqdm(finetune_loader) if local_rank == 0 else finetune_loader
        ep_samples = 0
        ep_correct = 0.0

        for frame, label in pbar:
            frame = frame.cuda(non_blocking=True).transpose(0, 1)   # T,N,C,H,W
            label = label.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                out_t = model(frame)          # shape: T,N,num_classes
                out   = out_t.mean(0)
                loss  = criterion(out, label)

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                ep_samples += label.size(0)
                ep_correct += (out.argmax(dim=1) == label).float().sum().item()
                functional.reset_net(model)

            if local_rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.6f}",
                    "acc" : f"{ep_correct / ep_samples:.4f}",
                })

    # ---------- validation (rank-0 only) ----------
    if local_rank == 0:
        model.eval()
        post_finetune = validate_model(
            model.module,                    # unwrap DDP
            T, test_ds, pruning_perc=f"{TARGET_PRUNE}_tuned",
            out_dir=MODEL_SAVE_PATH, batch=BATCH,
        )
        # Optionally save checkpoint
        print("post_finetune", post_finetune)

        inspect_layer_sparsity(flat)

    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
#  launcher
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # (Optional) extra CLI flags go here …
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 CUDA devices are required for DDP.")

    # torchrun sets LOCAL_RANK for each spawned process – honour it if present
    local_rank_env = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank_env >= 0:
        main(local_rank_env, args)
    else:
        # Fallback: manual spawn (rarely needed when using torchrun)
        torch.multiprocessing.spawn(
            main, args=(args,), nprocs=world_size, join=True
        )
