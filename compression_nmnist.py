from osbc_prune import *
from playgroundNMNIST import *

MODEL_SAVE_PATH = "./models/nmnist"

BATCH = 512

OSBS = False

TARGET_PRUNE = [0.50]

if __name__ == "__main__":
    # meta parameters
    tau = {
        1: 2.0,
        3: 2.0
    }
    T = 100
    model_path = "./models/nmnist/checkpoint_max.pth"  
    data_path = "./datas/NMNIST"

    model = SNN2L(tau=2.0).to(DEV)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEV)['net'])
    model.eval()
    print(model)
    
    lamps_score = LAMPS(model.layer)
    prune_percentage = get_prune_percs_from_lamps_single_target(lamps_score, target_perc=TARGET_PRUNE[-1])
    print(prune_percentage)

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on NMNIST train dataset, 
        # We will split test dataset into sample dataset and 
        # test dataset
     
        train_dataset = n_mnist.NMNIST(
            root=data_path,
            train=True,
            data_type='frame',
            frames_number=100,
            split_by='number',
        )

        test_dataset = n_mnist.NMNIST(
            root=data_path,
            train=False,
            data_type='frame',
            frames_number=100,
            split_by='number',
        )

        # split dataset [0.1, 0.9]. This gives us 1000 sample, 9000 test
        #quant_data, quant_test = split_neuromorphic_data_subsets(train_dataset, [0.1, 0.9])
        quant_data = train_dataset
        quant_test = test_dataset
        
        
        # validate model before training
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
            model.layer[name].weight.data = pruned_w
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