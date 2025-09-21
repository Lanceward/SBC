from osbc_prune import *
from playgroundCIFAR10DVS import *
from pathlib import Path
from torch.utils.data import Subset


MODEL_SAVE_PATH = "./models/cifar10dvs"

BATCH=32

OSBS = False

TARGET_PRUNE = [0.97]

if __name__ == "__main__":
    # meta parameters
    torch.manual_seed(TORCH_SEED)

    tau = {
        0: 2,
        4: 2,
        8: 2,
        12: 2,
        18: 2,
        21: 2,
    }
    T = 10
    model_path = "./models/cifar10dvs/checkpoint_max.pth"  
    data_path = "./datas/CIFAR10DVS"

    # Load trained weight to model
    # model = DVSCIFAR10NET_DOWNSIZED(channels=128, tau=2.0).to(DEV)
    model = DVSCIFAR10NET_FULLSIZED(channels=128, tau=2.0).to(DEV)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEV)['net'])
    model.eval()
    print(model)

    lamps_score = LAMPS(model.conv_fc)
    prune_percentage = get_prune_percs_from_lamps_single_target(lamps_score, target_perc=TARGET_PRUNE[-1])
    print(prune_percentage)

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on CIFAR10DVS train dataset, 
        # We will subsample train dataset for sample dataset and 
        # use whole test dataset for testing

        # data_transform = DVStransform(transforms.Resize(size=(64, 64), antialias=True))
        data_transform = None
        full_dataset = cifar10_dvs.CIFAR10DVS(
            root=data_path,
            data_type="frame",
            frames_number=T,
            split_by="number",
            transform=data_transform,
        )

        root_dir        = Path("cifar10dvs_data_64x64")              # keep split files in one folder
        root_dir.mkdir(exist_ok=True)

        train_idx_path  = root_dir / "train_idx.pt"
        test_idx_path   = root_dir / "test_idx.pt"

        # ------------------------------------------------------------------
        # 2.  Re-use split if it exists, otherwise create & persist it
        # ------------------------------------------------------------------
        if train_idx_path.exists() and test_idx_path.exists():
            # ---------- load ----------
            train_idx = torch.load(train_idx_path)
            test_idx  = torch.load(test_idx_path)

            train_dataset = Subset(full_dataset, train_idx)
            test_dataset  = Subset(full_dataset, test_idx)

            print(f"✔  Loaded cached split "
                f"({len(train_dataset)} train / {len(test_dataset)} test samples).")

        else:
            # ---------- create ----------
            train_dataset, test_dataset = split_to_train_test_set(
                train_ratio=0.9,
                origin_dataset=full_dataset,
                num_classes=10,
                random_split=False,
            )

            # ---------- save ----------
            torch.save(train_dataset.indices, train_idx_path)
            torch.save(test_dataset.indices,  test_idx_path)

            print(f"   Created new split and saved index files to ‘{root_dir}’\n"
                f"    ({len(train_dataset)} train / {len(test_dataset)} test samples).")

        quant_data, _ = split_neuromorphic_data_subsets(train_dataset, [0.2, 0.8])
        # quant_data = train_dataset
        quant_test = test_dataset

        pre_prune_loss = validate_model(model, T, quant_test, 0.0, out_dir=MODEL_SAVE_PATH, batch=BATCH)
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
            model.conv_fc[name].weight.data = pruned_w
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
        