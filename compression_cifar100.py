from osbc_prune import *
from playgroundCIFAR10DVS import *
from pathlib import Path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder, CIFAR100
from archs.cifarsvhn.vgg import vgg16_bn
import torchvision.transforms as transforms

BATCH=32

OSBS = False

TARGET_PRUNE = [0.9569]

MODEL_SAVE_PATH = "./models/cifar100_vgg"

class RepeatTransform:
    def __init__(self, T: int):
        self.T = T

    def __call__(self, x: torch.Tensor):
        # Input x has shape [C, H, W]
        return x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # -> [T, C, H, W]

def loadCIFAR100(data_path, T):
    """Return DataLoader for Cifar100 val/ subset."""
    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2673, 0.2564, 0.2762]
    tf_train_cifar100 = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD),
        RepeatTransform(T),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        RepeatTransform(T),
    ])
    dataset_train = CIFAR100(root=data_path, transform=tf_train_cifar100, train=True, download=True)
    dataset_test = CIFAR100(root=data_path, transform=valid_transform, train=False, download=True)
    
    return dataset_train, dataset_test

class FlattenedVGG(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        # Collect all submodules
        layers = []
        layers.extend(list(original_model.features.children()))
        layers.append(original_model.flatten)
        layers.append(original_model.classifier)

        # Create big Sequential
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    # meta parameters
    torch.manual_seed(TORCH_SEED)

    tau_singular = 4.0/3.0
    tau = {
        0: tau_singular,
        3: tau_singular,
        7: tau_singular,
        10: tau_singular,
        14: tau_singular,
        17: tau_singular,
        20: tau_singular,
        24: tau_singular,
        27: tau_singular,
        30: tau_singular,
        34: tau_singular,
        37: tau_singular,
        40: tau_singular,
        45: -1.0,
        #45: 1.0+1e-7, # no actual LIF module. Approximate OBC with small tau
    }
    T = 20
    model_path = "./models/cifar100_vgg/checkpoint_max_test_acc1.pth"  
    data_path = "./datas"

    # Load trained weight to model
    original_model = vgg16_bn(
        num_classes=100
    ).to(DEV)
    original_model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEV)['model'])
    model = FlattenedVGG(original_model).to(DEV).eval()
    print(model)
    
    lamps_score = LAMPS(model.layer, power=3.0)
    prune_percentage = get_prune_percs_from_lamps_single_target(lamps_score, target_perc=TARGET_PRUNE[-1])
    print(prune_percentage)

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on CIFAR10DVS train dataset, 
        # We will subsample train dataset for sample dataset and 
        # use whole test dataset for testing

        train_dataset, test_dataset = loadCIFAR100(data_path, T)
        _, test_datasetT5 = loadCIFAR100(data_path, 5)

        # quant_data, _ = split_neuromorphic_data_subsets(train_dataset, [0.1, 0.9])
        quant_data = train_dataset
        quant_test = test_dataset

        pre_prune_loss = validate_model(model, 5, test_datasetT5, 0.0, out_dir=MODEL_SAVE_PATH, batch=BATCH)
        print("pre_prune_loss", pre_prune_loss)
                
        start_time = time.time()
        accs, svs = optimal_spiking_brain_surgeon(model, quant_data, 
                                                tau=tau, T=T, prune_perc=prune_percentage, OSBS=OSBS,
                                                model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH)
        print("total time:" + "{:.2f}".format(time.time() - start_time))

        for key, value in accs.items():
            print(key, "  ", value)
        
        for name, sv in svs.items():
            pruned_w = sv.getWeightCopy()
            # assume all neurons in a single layer has the same tau
            model.layer[name].weight.data = pruned_w
        post_prune_loss = validate_model(model, 5, test_datasetT5, TARGET_PRUNE[-1], out_dir=MODEL_SAVE_PATH, batch=BATCH)
        
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
        