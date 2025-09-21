from osbc_quant import *

MODEL_SAVE_PATH = "./models/nmnist_quant"

BATCH = 1000 # Validation batch size. 

OSBS = True

TARGET_BIT_WIDTH = 4

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
    model.load_state_dict(torch.load(model_path, weights_only=False)['net'])
    model.eval()
    print(model)
    
    print("target bit width: ", str(TARGET_BIT_WIDTH))

    # we record layer input and ouput in this block
    with torch.no_grad():
        # Load sample dataset and test dataset.
        # Since the sample is trained on NMNIST train dataset, 
        # We will split test dataset into sample dataset and 
        # test dataset

        test_dataset = n_mnist.NMNIST(
            root=data_path,
            train=False,
            data_type='frame',
            frames_number=100,
            split_by='number',
        )

        quant_accuracies = []
        prequant_accuracies = []

        # 5 runs, take the average
        for torch_seeds in TORCH_SEEDS:
            model.load_state_dict(torch.load(model_path, weights_only=False)['net'])
            model.eval()
            
            #1000 sample 9000 test
            quant_data_batch, quant_data_labels_batch, quant_test, quant_test_labels = split_neuromorphic_data_stratified(test_dataset, [0.1, 0.9], torch_seeds)
            print(quant_data_batch.shape, quant_data_labels_batch.shape, quant_test.shape, quant_test_labels.shape)        
            
            pre_prune_loss = validate_model(model, T, quant_test, quant_test_labels, bitw=32, out_dir=MODEL_SAVE_PATH, batch=BATCH)
            print("pre_prune_loss", pre_prune_loss)
            prequant_accuracies.append(pre_prune_loss)
            
            start_time = time.time()
            svs = optimal_spiking_brain_quantizer(model, quant_data_batch, 
                                                    tau=tau, T=T, bit_width=TARGET_BIT_WIDTH, OSBS=OSBS,
                                                    validation_data=quant_test, validation_label=quant_test_labels, model_output_dir=MODEL_SAVE_PATH, validation_batch_size=BATCH)
            print("total time:" + "{:.2f}".format(time.time() - start_time))

            for name, sv in svs.items():
                quant_w = sv.getWeightCopy()
                model.layer[name].weight.data = quant_w
            post_prune_loss = validate_model(model, T, quant_test, quant_test_labels, bitw="seed_"+str(torch_seeds)+"_"+str(TARGET_BIT_WIDTH), out_dir=MODEL_SAVE_PATH, batch=BATCH)
            
            quant_accuracies.append(post_prune_loss)
            print("post_prune_loss", post_prune_loss)
            
        print("Pre quantization: " + str(prequant_accuracies) + ", Post quantization: " + str(quant_accuracies), np.mean(quant_accuracies), np.mean(prequant_accuracies))
