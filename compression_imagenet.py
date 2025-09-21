from playgroundImageNet import *
from osbc_prune import *
from pathlib import Path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder, CIFAR100
from archs.cifarsvhn.vgg import vgg16_bn
import torchvision.transforms as transforms

BATCH=32

OSBS = True

TARGET_PRUNE = [0.7892]

MODEL_SAVE_PATH = "./models/imagenet"

class RepeatTransform:
    def __init__(self, T: int):
        self.T = T

    def __call__(self, x: torch.Tensor):
        # Input x has shape [C, H, W]
        return x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # -> [T, C, H, W]

def loadImageNet(data_path, T):
    """Return DataLoader for Cifar100 val/ subset."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    crop_size = 224
    resize_size = 256
    interpolation = InterpolationMode.BILINEAR
    tf_train_imagenet = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
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

class FlattenedSEWResNet(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        # Collect all submodules
        layers = []
        layers.append(original_model.conv1)
        layers.append(original_model.bn1)
        layers.append(original_model.sn1)
        layers.append(original_model.maxpool)
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer1.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer2.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer3.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer4.children())))
        layers.append(original_model.avgpool)
        layers.append(original_model.fc)

        # Create big Sequential
        self.layer = nn.Sequential(*layers)

    def _flatten_BasicBlocks(self, basicblock_list):
        bb_flat = []
        for bb in basicblock_list:
            # 1st conv module
            bb_flat.append(bb.conv1)
            bb_flat.append(bb.bn1)
            bb_flat.append(bb.sn1)
            # 2nd conv module
            bb_flat.append(bb.conv2)
            bb_flat.append(bb.bn2)
            bb_flat.append(bb.sn2)
            # skip conneciton. It is flattened here
            # because FlattenedResnetLayers not real
            # model, but a structure where OSBC can parse
            if bb.downsample is not None:
                bb_flat.extend(list(bb.downsample.children()))
                bb_flat.append(bb.downsample_sn)
        return bb_flat

    def forward(self, x):
        return self.layer(x)

class FlattenedSEWResNetBottleNeck(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        # Collect all submodules
        layers = []
        layers.append(original_model.conv1)
        layers.append(original_model.bn1)
        layers.append(original_model.sn1)
        layers.append(original_model.maxpool)
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer1.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer2.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer3.children())))
        layers.extend(self._flatten_BasicBlocks(list(original_model.layer4.children())))
        layers.append(original_model.avgpool)
        layers.append(original_model.fc)

        # Create big Sequential
        self.layer = nn.Sequential(*layers)

    def _flatten_BasicBlocks(self, basicblock_list):
        bb_flat = []
        for bb in basicblock_list:
            # 1st conv module
            bb_flat.append(bb.conv1)
            bb_flat.append(bb.bn1)
            bb_flat.append(bb.sn1)
            # 2nd conv module
            bb_flat.append(bb.conv2)
            bb_flat.append(bb.bn2)
            bb_flat.append(bb.sn2)
            # 3rd conv module
            bb_flat.append(bb.conv3)
            bb_flat.append(bb.bn3)
            bb_flat.append(bb.sn3)
            # skip conneciton. It is flattened here
            # because FlattenedResnetLayers not real
            # model, but a structure where OSBC can parse
            if bb.downsample is not None:
                bb_flat.extend(list(bb.downsample.children()))
                bb_flat.append(bb.downsample_sn)
        return bb_flat

    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    # meta parameters
    torch.manual_seed(TORCH_SEED)

    T = 4
    model_path = "./models/imagenet/sew18_checkpoint_319.pth"
    data_path = "../datasets/ImageNet"

    # Load trained weight to model
    """Return a spiking ResNet19 BN with ImageNet weights already loaded."""
    model = sew_resnet.sew_resnet18(
        pretrained=False,
        cnf='ADD',                                 # 'ADD', 'AND', or 'IAND'
        spiking_neuron=neuron.IFNode,     # Integrate‑and‑fire
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(),       # surrogate gradient
        detach_reset=False,
    ).to(DEV).eval()
    # # 1. Load the raw checkpoint
    ckpt = torch.load(model_path, map_location=DEV, weights_only=False)
    ckpt = ckpt['model']
    sd_mapped = RemapSEWResNetCheckpoints(ckpt)
    # 4. Load weights (conv/bn layers will match exactly)
    missing, unexpected = model.load_state_dict(sd_mapped, strict=False)
    print('Missing keys (will be neuron-only):', missing)
    print('Unexpected keys:', unexpected)
    # model_flatten_layers = FlattenedSEWResNetBottleNeck(model).to(DEV).eval()
    model_flatten_layers = FlattenedSEWResNet(model).to(DEV).eval()
    print(model_flatten_layers)
    
    lamps_score = LAMPS(model_flatten_layers.layer, power=3.0)
    prune_percentage = get_prune_percs_from_lamps_single_target(lamps_score, target_perc=TARGET_PRUNE[-1])
    print(prune_percentage)

    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model_flatten_layers.parameters()))

    # construct tau list for all prunable layers
    tau_singular = 1e7
    tau = {x: tau_singular for x in prune_percentage}
    tau[62] = -1.0

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on CIFAR10DVS train dataset, 
        # We will subsample train dataset for sample dataset and 
        # use whole test dataset for testing

        train_dataset, test_dataset = loadImageNet(data_path, T)

        quant_data, _ = stratified_split(train_dataset, frac=0.02)
        # quant_data = train_dataset
        # quant_test, _ = split_neuromorphic_data_subsets(test_dataset, [0.1, 0.9])
        quant_test = test_dataset

        print(len(quant_data), len(quant_test))
        
        pre_prune_loss = 0.6918#validate_model(model, T, quant_test, 0.0, out_dir=MODEL_SAVE_PATH, batch=BATCH)
        print("pre_prune_loss", pre_prune_loss)
        
        start_time = time.time()
        accs, svs = optimal_spiking_brain_surgeon(model_flatten_layers, quant_data, 
                                                tau=tau, T=T, prune_perc=prune_percentage, OSBS=OSBS,
                                                model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH, real_model=model)
        print("total time:" + "{:.2f}".format(time.time() - start_time))

        for key, value in accs.items():
            print(key, "  ", value)
        
        for name, sv in svs.items():
            pruned_w = sv.getWeightCopy()
            # assume all neurons in a single layer has the same tau
            model_flatten_layers.layer[name].weight.data = pruned_w
        post_prune_loss = validate_model(model, T, quant_test, TARGET_PRUNE[-1], out_dir=MODEL_SAVE_PATH, batch=BATCH)
        
        print("Pre prune: " + str(pre_prune_loss) + ", Post prune: " + str(post_prune_loss))
        
        # validate that X percentage of weights are indeed pruned
        pruend_weight = 0
        total_weight = 0
        for name, sv in svs.items():
            pruend_weight += (torch.sum(torch.abs(sv.W) < 5e-7)).item()
            total_weight += sv.W.numel()
            pruned_percentage = (torch.sum(torch.abs(sv.W) < 5e-7) / sv.W.numel()).item()
            print(str(name) + " pruned percentage: " + str(pruned_percentage * 100))
        print("total pruned percentage: " + str(pruend_weight/total_weight * 100))
    print(sum(p.numel() for p in model.parameters()))
