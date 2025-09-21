from osbc_quant import *
from playgroundDVS128 import *

MODEL_SAVE_PATH = "./models/dvs128gesture_quant"

BATCH=32 # Validation batch size. 

OSBS = True

TARGET_BIT_WIDTH = 4

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
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cuda')['net'])
    model.eval()
    print(model)
    
    print("target bit width: ", str(TARGET_BIT_WIDTH))

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

        _, _, quant_test, quant_test_labels = split_neuromorphic_data(test_dataset, [0.0, 1.0])
        # validate model before pruning
        pre_prune_loss = validate_model(model, T, quant_test, quant_test_labels, bitw=32, out_dir=MODEL_SAVE_PATH, batch=BATCH)
        print("pre_prune_loss", pre_prune_loss)

        quant_accuracies = []

        for torch_seeds in TORCH_SEEDS:
            model.load_state_dict(torch.load(model_path, weights_only=False)['net'])
            model.eval()
            # split dataset. This gives us 200 sample, 288 test
            quant_data_batch, quant_data_labels_batch, _, _ = split_neuromorphic_data_stratified(train_dataset, [0.2, 0.8], torch_seeds)
            print(quant_data_batch.shape, quant_data_labels_batch.shape, quant_test.shape, quant_test_labels.shape)        
            
            print(torch.bincount(quant_data_labels_batch.cpu(), minlength=torch.max(quant_data_labels_batch.cpu())+1))
            
            start_time = time.time()
            svs = optimal_spiking_brain_quantizer(model, quant_data_batch, 
                                                    tau=tau, T=T, bit_width=TARGET_BIT_WIDTH, OSBS=OSBS,
                                                    validation_data=quant_test, validation_label=quant_test_labels, model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH)
            print("total time:" + "{:.2f}".format(time.time() - start_time))
                
            for name, sv in svs.items():
                quant_w = sv.getWeightCopy()
                model.conv_fc[name].weight.data = quant_w
            post_prune_loss = validate_model(model, T, quant_test, quant_test_labels, bitw="seed_"+str(torch_seeds)+"_"+str(TARGET_BIT_WIDTH), out_dir=MODEL_SAVE_PATH, batch=BATCH)
            
            quant_accuracies.append(post_prune_loss)
            
            print(post_prune_loss)
        
        print("Pre quant: " + str(pre_prune_loss) + ", Post quant: " + str(quant_accuracies), np.mean(quant_accuracies))