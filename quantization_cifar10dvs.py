from osbc_quant import *
from playgroundCIFAR10DVS import *
from pathlib import Path
from torch.utils.data import Subset

MODEL_SAVE_PATH = "./models/cifar10dvs_quant"

BATCH=256 # Validation batch size. 

OSBS = True

TARGET_BIT_WIDTH = 4

if __name__ == "__main__":
    # meta parameters
    torch.manual_seed(TORCH_SEED)

    tau = {
        0: 2.0,
        4: 2.0,
        8: 2.0,
        12: 2.0,
        18: 2.0,
        21: 2.0,
    }
    T = 20
    model_path = "./models/cifar10dvs/checkpoint_max.pth"  
    data_path = "./datas/CIFAR10DVS"

    # Load trained weight to model
    model = DVSCIFAR10NET_DOWNSIZED(channels=128, tau=2.0).to(DEV)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cpu')['net'])
    model = model.to(DEV)
    model.eval()
    #print(model)

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on CIFAR10DVS train dataset, 
        # We will subsample train dataset for sample dataset and 
        # use whole test dataset for testing

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

        # root_dir        = Path("cifar10dvs_data_64x64")              # keep split files in one folder
        root_dir        = Path("cifar10dvs_data_128x128")              # keep split files in one folder
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


        # _, _, quant_test, quant_test_labels = split_neuromorphic_data(test_dataset, [0.0, 1.0])
        #quant_data, _ = split_neuromorphic_data_subsets(train_dataset, [0.1, 0.9])
        quant_data = train_dataset
        quant_test = test_dataset


        # validate model before pruning
        pre_prune_loss = validate_model(model, T, quant_test, bitw=32, out_dir=MODEL_SAVE_PATH, batch=BATCH)
        print("pre_prune_loss", pre_prune_loss)        
        
        quant_accuracies = []

        for torch_seeds in TORCH_SEEDS:
            model.load_state_dict(torch.load(model_path, weights_only=False)['net'])
            model.eval()
            # split dataset. This gives us 900 sample, 1000 test
            quant_data_batch, quant_data_labels_batch, _, _ = split_neuromorphic_data_stratified(train_dataset, [0.1, 0.9], torch_seeds)
            print(torch.bincount(quant_data_labels_batch.cpu(), minlength=torch.max(quant_data_labels_batch.cpu())+1))

            start_time = time.time()
            svs = optimal_spiking_brain_quantizer(model, quant_data, 
                                                    tau=tau, T=T, bit_width=TARGET_BIT_WIDTH, OSBS=OSBS,
                                                    model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH)
            print("total time:" + "{:.2f}".format(time.time() - start_time))
                
            for name, sv in svs.items():
                quant_w = sv.getWeightCopy()
                model.conv_fc[name].weight.data = quant_w
            post_prune_loss = validate_model(model, T, quant_test, bitw="seed_"+str(torch_seeds)+"_"+str(TARGET_BIT_WIDTH), out_dir=MODEL_SAVE_PATH, batch=BATCH)
            
            quant_accuracies.append(post_prune_loss)
            
            print(post_prune_loss)
            
        print("Pre quantization: " + str(pre_prune_loss) + ", Post quantization: " + str(quant_accuracies), np.mean(quant_accuracies))
