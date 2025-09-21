from osbc_prune import *
from playgroundDVS128 import *

MODEL_SAVE_PATH = "./models/dvs128gesture"

BATCH=32

OSBS = True

TARGET_PRUNE = [0.98]

if __name__ == "__main__":
    # meta parameters
    torch.manual_seed(TORCH_SEED)

    tau = {
        0: 2,
        4: 2,
        8: 2,
        12: 2,
        16: 2,
        22: 2,
        25: 2
    }
    T = 20
    model_path = "./models/dvs128gesture/checkpoint_max.pth"  
    data_path = "./datas/DVS128Gesture"

    # Load trained weight to model
    model = DVSGestureNetParametric(channels=128, tau=2.0).to(DEV)
    model.load_state_dict(torch.load(model_path, weights_only=False)['net'])
    model.eval()
    print(model)

    lamps_score = LAMPS(model.conv_fc)
    prune_percentage = get_prune_percs_from_lamps(lamps_score, target_percentages=TARGET_PRUNE)
    print(prune_percentage)
    
    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on DVS128Gesture train dataset, 
        # We will subsample train dataset for sample dataset and 
        # use whole test dataset for testing

        train_dataset = dvs128_gesture.DVS128Gesture(
            root=data_path,
            train=True,
            data_type='frame',
            frames_number=T,
            split_by='number',
        )
        test_dataset = dvs128_gesture.DVS128Gesture(
            root=data_path,
            train=False,
            data_type='frame',
            frames_number=T,
            split_by='number',
        )

        # split dataset. This gives us 200 sample, 288 test
        quant_data_batch, quant_data_labels_batch, _, _ = split_neuromorphic_data(train_dataset, [0.2, 0.8])
        _, _, quant_test, quant_test_labels = split_neuromorphic_data(test_dataset, [0.0, 1.0])
        print(quant_data_batch.shape, quant_data_labels_batch.shape, quant_test.shape, quant_test_labels.shape)        

        # validate model before pruning
        pre_prune_loss = validate_model(model, T, quant_test, quant_test_labels, 0.0, out_dir=MODEL_SAVE_PATH, batch=BATCH)
        print("pre_prune_loss", pre_prune_loss)
        
        start_time = time.time()
        accs, svs = optimal_spiking_brain_surgeon(model, quant_data_batch, 
                                                tau=tau, T=T, prune_perc=prune_percentage, OSBS=OSBS,
                                                validation_data=quant_test, validation_label=quant_test_labels, model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH)
        print("total time:" + "{:.2f}".format(time.time() - start_time))

        for key, value in accs.items():
            print(key, "  ", value)
            
        for name, sv in svs.items():
            pruned_w = sv.getWeightCopy()
            model.conv_fc[name].weight.data = pruned_w
        post_prune_loss = validate_model(model, T, quant_test, quant_test_labels, 0.999)
        
        print("Pre prune: " + str(pre_prune_loss) + ", Post prune: " + str(post_prune_loss))
        
        # validate that X percentage of weights are indeed pruned
        for name, sv in svs.items():
            pruned_percentage = (torch.sum(torch.abs(sv.W) < 5e-6) / sv.W.numel()).item()
            print(str(name) + " pruned percentage: " + str(pruned_percentage * 100))
        
        
