# In this script, I will try to apply Optimal Brain Surgeon for Spiking Neural Network
# I will be doing per layer Pruning, i.e. quantizing weights to 0
# Proofs of algorithm can be seen in the two pdfs I put in here

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from spikingjelly import activation_based
from spikingjelly.datasets import n_mnist, transforms, dvs128_gesture
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# from quant_models import *
from quantizer import *
from modelutils import *

RTN = False

class SurrogateVoltage:
    #Implementation assumes each pass contains 1 data sample
    def __init__(self, nname, llayer, data_length, tau, T, spacial_size = [], is_osbs = True, batchnormlayer = None, wbit = 16):
        self.name = str(nname)
        self.layer = llayer
        self.dev = self.layer.weight.device
        self.tau = tau
        self.T = T
        self.data_count = 0
        self.data_length = data_length
        self.linearized_data_count = 0
        self.is_osbs = is_osbs
        self.quantizer = Quantizer()
        if batchnormlayer is not None:
            self.hasbatchnorm = True
            bn = batchnormlayer
            # input X of shape (C, H, W):
            gamma = bn.weight.data # (C)
            var = bn.running_var # (C)
            eps = bn.eps # (1)
            # Compute per-channel constants
            self.AFF = gamma / torch.sqrt(var + eps) # (C)
        else:
            self.hasbatchnorm = False
        
        if isinstance(self.layer, layer.Linear):
            #parameters
            self.linearized_in_features = self.layer.in_features
            self.linearized_out_features = self.layer.out_features
            
            #init weight
            self.W = self.layer.weight.clone()
            # print(self.W.shape)        
            self.quantizer.configure(bits=wbit, perchannel=True, sym=True, mse=False, norm=2.0)
        elif isinstance(self.layer, layer.Conv2d):
            spacial_size = torch.tensor(spacial_size, dtype=torch.int)
            #parameters
            self.unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            self.spacial_size = spacial_size
            C = self.layer.in_channels
            prod_in_kernel = self.layer.kernel_size[0] * self.layer.kernel_size[1]
            self.linearized_in_features = C * prod_in_kernel
            self.linearized_out_features = self.layer.out_channels

            # print(self.layer.weight.shape)
            self.W = self.layer.weight.clone().flatten(1) #(out_channel, in_channel, kernel_size0, kernel_size1) -> (out_channel, in_channel * kernel_size0 * kernel_size1)
            self.quantizer.configure(bits=wbit, perchannel=True, sym=True, mse=False)
        else:
            raise ValueError(type(self.layer) + " unsupported")

        print("    layer summary: tau:", self.tau, "T", self.T, "Linearized parameters:", self.linearized_in_features, self.linearized_out_features, "bit width:", wbit)

        self.H = torch.zeros((self.linearized_in_features, self.linearized_in_features)).to(self.dev) # Hessian

        #build convolution matrix
        self.conv = torch.zeros((T, T)).to(self.dev)
        conv_row = torch.zeros((1, T)).to(self.dev)
        for t in range(T):
            conv_row[0, t] = 1/self.tau * ((self.tau - 1)/self.tau)**(T-t-1)
        for i in range(T):
            self.conv[i, :i+1] = conv_row[0, T-i-1:]
        
        print(self.W.shape)
        self.quantizer.find_params(self.W, weight=True)
        #print(self.quantizer.maxq, self.quantizer.zero, self.quantizer.scale, self.quantizer.maxq)
    
    # Add 1 sample worth of data. 
    # Assume input has size (T, 1, layer.in_features)
    # Assume output has size (T, 1, layer.out_features)
    def add_data_sample(self, inp, out):
        assert inp[0].shape[1] == 1 and out.shape[1] == 1 # assert theres only 1 input
        if isinstance(self.layer, layer.Linear):
            self.linearized_data_count += 1
            inp = inp[0][:, 0, :] #(T, 1, in) -> (T, in_features)
            if self.is_osbs:
                KS = torch.matmul(self.conv, inp) #(T, T) x (T, in_features) -> (T, in_features)
            else:
                KS = inp # (T, in_features)
        elif isinstance(self.layer, layer.Conv2d):
            inp = inp[0][:, 0, :, :, :] #(T, 1, C, H, W) -> (T, C, H, W)
            if self.is_osbs: 
                inp = torch.einsum('ti,ichw->tchw', self.conv, inp) #(T, T) x (T, C, H, W) -> (T, C, H, W)
            inp = self.unfold(inp) #(T, C, H, W) -> (T, C * kernel0 * kernel1, L)
            self.linearized_data_count += inp.shape[2]
            inp = inp.permute([1, 0, 2]) #(T, C * kernel0 * kernel1, L) -> (C * kernel0 * kernel1, T, L)
            KS = inp.flatten(1) #(C * kernel0 * kernel1, T, L) -> (C * kernel0 * kernel1, T * L)
            KS = KS.t() # (C * kernel0 * kernel1, T * L) -> (T * L, C * kernel0 * kernel1)
        else:
            raise ValueError(type(self.layer) + " unsupported")
        
        self.data_count += 1
                
        self.H += 2 * KS.t().matmul(KS)
        
        print("\rlayer hessian progress: " + "{:.2f}".format(self.data_count/self.data_length * 100) + "%", end='')

    def quantize(self):
        # assume all data samples have been added
        assert self.data_count == self.data_length and self.H.shape[0] == self.linearized_in_features
        print("Layer " + self.name + " data length: " + str(self.data_count) + ", linearized data length: " + str(self.linearized_data_count))

        #self.H = self.H.unsqueeze(0).repeat(self.linearized_out_features, 1, 1).to(DEV)

        # # add batchnorm
        # if self.hasbatchnorm and self.is_osbs:
        #     self.H = self.H * (self.AFF.view(-1, 1, 1) * self.AFF.view(-1, 1, 1) / self.linearized_data_count)
        # else:  
        self.H /= self.linearized_data_count

        #avg_H =torch.mean(self.H, dim=0)
        #print(avg_H, torch.mean(avg_H), torch.var(avg_H))

        # for numerical stability
        percdamp = 0.01
        #damp = percdamp * torch.mean(torch.diagonal(self.H, dim1=1, dim2=2), dim=1)
        damp = percdamp * torch.mean(torch.diagonal(self.H, dim1=0, dim2=1))
        diag = torch.arange(self.linearized_in_features, device=self.dev)
                
        self.H[diag, diag] += damp
                
        # calculate inverse hessian
        self.H_invs = torch.inverse(self.H.to(torch.device('cpu'))).to(DEV)#torch.zeros_like(self.H)
        torch_empty_cache()
        
        # !! FAST & MEMORY INTENSIVE !!
        # Calculate the pruning order of weights per channel
        # temporary parameters to do pruning on
        clock_ordering = time.time()

        # because each channel has a seperate hessian that does not
        # interfere with each other, we can do the quantization of
        # all the channels in parallel
        channels = torch.arange(self.linearized_out_features, dtype=torch.int, device=self.dev)#torch.tensor(channels_not_zeros, dtype=torch.int, device=self.dev)
        
        # # get quant order
        # if isinstance(self.layer, layer.Linear):
        avg_Hinv_diags = torch.diag(self.H_invs)#torch.mean(torch.diagonal(self.H_invs, dim1=1, dim2=2)[channels], dim=0)
        # else:
        #avg_Hinv_diags = torch.arange(self.linearized_in_features)#torch.mean(torch.diagonal(self.H_invs, dim1=1, dim2=2)[channels], dim=0)
        quant_order = torch.argsort(avg_Hinv_diags)
        
        temp_W = self.W.clone()
        temp_Hinv = self.H_invs.clone()
        temp_quant_mask = torch.ones_like(self.W, dtype=bool, device=self.dev)
        temp_quant_mask[channels] = False
        # for widx in range(self.linearized_in_features):
        #     print("\rweight quantization progress: " + "{:.2f}".format(widx/self.linearized_in_features * 100) + "% " + str(torch.sum(~temp_quant_mask)), end='')
        #     if not RTN:
        #         widxs = quant_order[widx]

        #         #figure out what to quant
        #         # for each row, earlist unquant row element, unless any element is out of bound
        #         wqs = self.quantizer.quantize_unclamp(temp_W)
        #         q_diff = torch.abs(wqs - temp_W)

        #         print(torch.sum((q_diff > self.quantizer.scale/2)[channels]), end='')

        #         wqs = wqs[channels, widxs]
                                
        #         # quntize and compensate weights
        #         cw = (((wqs-temp_W[channels, widxs]) / temp_Hinv[widxs, widxs]).unsqueeze(1) * temp_Hinv[:, widxs].unsqueeze(0))

        #         cw[temp_quant_mask[channels]] = 0
        #         temp_W.index_add_(0, channels, cw)
        #         temp_quant_mask[channels, widxs] = True
                
        #         #compensate hessian. Lemma 1 from paper:
        #         # Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning
        #         Hc = temp_Hinv               # shape (I, I)
        #         pivot = Hc[widxs, widxs]                   # scalar H_c[w,w]
        #         u = Hc[:, widxs]                       # shape (I,)
        #         v = Hc[widxs, :]                       # shape (I,)

        #         # rank-1 update: Hc ← Hc − (1/pivot) * (u outer v)
        #         temp_Hinv -= (1.0 / pivot) * torch.outer(u, v)

        #     else:
        #         temp_W[:, widx] = self.quantizer.quantize(temp_W[:, widx].unsqueeze(1)).squeeze(1)
        
        # print("finished. Took " + "{:.2f}".format((time.time() - clock_ordering)) + "s. Average " + "{:.2f}".format((self.linearized_in_features*self.linearized_out_features) / (time.time() - clock_ordering)) + "weights/s")
        
        oldscale = self.quantizer.scale.clone()
        
        self.quantizer.find_params(temp_W, weight=True)
        scalediff = torch.abs(oldscale - self.quantizer.scale)
        print(torch.min(scalediff), torch.max(scalediff))

        quant_mask = torch.ones_like(self.W, dtype=bool, device=self.dev)
        quant_mask[channels] = False
        for widx in range(self.linearized_in_features):
            print("\rweight quantization progress: " + "{:.2f}".format(widx/self.linearized_in_features * 100) + "% " + str(torch.sum(~quant_mask)), end='')
            if not RTN:
                widxs = quant_order[widx]

                #figure out what to quant
                # for each row, earlist unquant row element, unless any element is out of bound
                wqs = self.quantizer.quantize(self.W)
                q_diff = torch.abs(wqs - self.W)
                print(torch.sum((q_diff > self.quantizer.scale/2)[channels]), end='')

                wqs = wqs[channels, widxs]
                                
                # quntize and compensate weights
                cw = (((wqs-self.W[channels, widxs]) / self.H_invs[widxs, widxs]).unsqueeze(1) * self.H_invs[:, widxs].unsqueeze(0))

                cw[quant_mask[channels]] = 0
                self.W[channels] += cw
                #self.W.index_add_(0, channels, cw)
                quant_mask[channels, widxs] = True
                
                #compensate hessian. Lemma 1 from paper:
                # Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning
                Hc = self.H_invs               # shape (I, I)
                pivot = Hc[widxs, widxs]                   # scalar H_c[w,w]
                u = Hc[:, widxs]                       # shape (I,)
                v = Hc[widxs, :]                       # shape (I,)

                # rank-1 update: Hc ← Hc − (1/pivot) * (u outer v)
                self.H_invs -= (1.0 / pivot) * torch.outer(u, v)
                
                #print(torch.sum(self.H_invs[:, widxs]), torch.sum(self.H_invs[widxs, :]))

            else:
                self.W[:, widx] = self.quantizer.quantize(self.W[:, widx].unsqueeze(1)).squeeze(1)
                
        print("finished. Took " + "{:.2f}".format((time.time() - clock_ordering)) + "s. Average " + "{:.2f}".format((self.linearized_in_features*self.linearized_out_features) / (time.time() - clock_ordering)) + "weights/s")

    def getWeightCopy(self):
        qws = self.quantizer.quantize(self.W)
        diff = torch.mean((qws - self.W).pow(2))
        print("mean squared", diff)
        self.W = qws
        if isinstance(self.layer, layer.Linear):
            w = self.W.clone()
            return self.W.clone()
        elif isinstance(self.layer, layer.Conv2d):
            flattened_w = self.W.clone()
            w = flattened_w.unflatten(1, (self.layer.in_channels, self.layer.kernel_size[0], self.layer.kernel_size[1]))
            return w
        else:
            raise ValueError(type(self.layer) + "Not supported")

# define hook
def record_io(sv):
    def tmp(_, inp, out):
        sv.add_data_sample(inp, out)
    return tmp

def optimal_spiking_brain_quantizer(model, input_sample_dataset, tau: dict, T, bit_width, validation_data, validation_label, model_output_dir, OSBS, validation_batch_size):
    data, _ = input_sample_dataset[0]
    sample_dataset_shape = [len(input_sample_dataset), data.shape[0], data.shape[1], data.shape[2], data.shape[3]] 

    sample_dataset_loader = torch.utils.data.DataLoader(input_sample_dataset, batch_size=16, shuffle=True,
                                                num_workers=8, pin_memory=True, sampler=None, 
                                                persistent_workers=False, prefetch_factor=2)
    
    # assume all neurons in a single layer has the same tau
    if hasattr(model, "conv_fc"):
        model_layer = model.conv_fc
    elif hasattr(model, "layer"):
        model_layer = model.layer
    else:
        raise ValueError("Model not supported!")
    
    full_set = find_layers(model_layer) # find all layers we want to record IO
    batchnorm_layers = find_layers(model_layer, layers=[activation_based.layer.BatchNorm2d])
    # assert all layers have tau value
    for name, layer in full_set.items():
        assert tau[name] != None

    svs = {}
    handles = []
    for name, layer in full_set.items():
        print("Name:", name, layer)
        if name+1 in batchnorm_layers:
            # if the following layer is batchnorm
            sv = SurrogateVoltage(name, layer, input_sample_batch.shape[0], tau[name], T, spacial_size=[input_sample_batch.shape[3], input_sample_batch.shape[4]], is_osbs=OSBS, batchnormlayer = batchnorm_layers[name+1], wbit=bit_width)
        else:
            # create surrogate voltage calculator
            sv = SurrogateVoltage(name, layer, input_sample_batch.shape[0], tau[name], T, spacial_size=[input_sample_batch.shape[3], input_sample_batch.shape[4]], is_osbs=OSBS, wbit=bit_width)
        handles.append(layer.register_forward_hook(record_io(sv)))
        svs[name] = sv

    functional.set_step_mode(model, 'm')

    #calculate new inputs for layer to be pruned
    for i in range(input_sample_batch.shape[0]):
        functional.reset_net(model)
        input_t = input_sample_batch[i, :, :, :, :]
        input_t = torch.unsqueeze(input_t, 1)
        model(input_t)
    functional.reset_net(model)

    #remove hook and unnecessary data
    for h in handles:
        h.remove()
    del input_sample_batch
    torch_empty_cache()

    # Initialize all SV modules
    for name, sv in svs.items():
        sv.quantize()
        torch_empty_cache()
        
    return svs

def validate_model(model, T, validation_dataset, bitw, out_dir = "", batch = 32):
    model_name = str(bitw) + "bit_quantized_model.pth"    
    functional.set_step_mode(model, 'm')
    total_correct = 0

    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch, shuffle=True,
                                                num_workers=8, pin_memory=True, sampler=None, 
                                                persistent_workers=False, prefetch_factor=4)

    for validation_data, label in tqdm(validation_data_loader):
        #Return model accuracy on inputs and labels
        validation_data = validation_data.to(DEV)
        validation_data = validation_data.transpose(0, 1)
        label = label.to(DEV)
        out_fr = model(validation_data)
        outs = out_fr.mean(0)
        total_correct += (outs.argmax(dim=1) == label).float().sum().item()
        functional.reset_net(model)
        
    functional.reset_net(model)

    if out_dir != "":
        torch.save(model, os.path.join(out_dir, model_name))

    return total_correct/len(validation_dataset)

# (B, T, C, W, H)
def split_image_data(dataset, encoder, split = [0.2, 0.8], T = 100):
    # split dataset [0.2, 0.8]. This gives us 1000 sample, 8000 test
    torch.manual_seed(TORCH_SEED)
    quant_data, quant_test = data.random_split(dataset, split)
    
    # batch quant_data
    quant_data_indices = quant_data.indices
    quant_data_samples = [dataset[i][0] for i in quant_data_indices]
    quant_data = torch.stack(quant_data_samples).to(torch.device(DEV))
    quant_data_labels = [dataset[i][1] for i in quant_data_indices]
    quant_data_labels = [dataset[i][1] for i in quant_data_indices]
    quant_data_labels = torch.tensor(quant_data_labels).to(torch.device(DEV))
    
    # batch test data
    quant_test_indices = quant_test.indices
    quant_test = [dataset[i][0] for i in quant_test_indices]
    quant_test = torch.stack(quant_test).to(torch.device(DEV))
    quant_test_labels = [dataset[i][1] for i in quant_test_indices]
    quant_test_labels = torch.tensor(quant_test_labels).to(torch.device(DEV))
    
    torch.manual_seed(TORCH_SEED)
    quant_data_batch = torch.zeros_like(quant_data).unsqueeze(1).repeat(1, T, 1, 1, 1)
    for t in range(T):
        quant_data_batch[:, t] = encoder(quant_data)

    torch.manual_seed(TORCH_SEED)
    quant_test_batch = torch.zeros_like(quant_test).unsqueeze(1).repeat(1, T, 1, 1, 1)
    for t in range(T):
        quant_test_batch[:, t] = encoder(quant_test)
        
    return quant_data_batch, quant_data_labels, quant_test_batch, quant_test_labels

# (B, T, C, W, H)
def split_neuromorphic_data(dataset, split = [0.2, 0.8], random_seed = TORCH_SEED):
    # split dataset [0.2, 0.8]. This gives us 1000 sample, 8000 test
    generator = torch.Generator().manual_seed(random_seed)
    quant_data, quant_test = data.random_split(dataset, split, generator=generator)
    
    if len(quant_data) != 0:
        # batch quant_data
        quant_data_indices = quant_data.indices
        quant_data_samples = [torch.tensor(dataset[i][0]) for i in quant_data_indices]
        quant_data = torch.stack(quant_data_samples).to(torch.device(DEV))
        quant_data_labels = [dataset[i][1] for i in quant_data_indices]
        quant_data_labels = torch.tensor(quant_data_labels).to(torch.device(DEV))
    else:
        quant_data, quant_data_labels = torch.empty(0), torch.empty(0)

    if len(quant_test) != 0:
        # batch test data
        quant_test_indices = quant_test.indices
        quant_test = [torch.tensor(dataset[i][0]) for i in quant_test_indices]
        quant_test = torch.stack(quant_test).to(torch.device(DEV))
        quant_test_labels = [dataset[i][1] for i in quant_test_indices]
        quant_test_labels = torch.tensor(quant_test_labels).to(torch.device(DEV))
    else:
        quant_test, quant_test_labels = torch.empty(0), torch.empty(0)

    return quant_data, quant_data_labels, quant_test, quant_test_labels

def split_neuromorphic_data_stratified(dataset, split=[0.2, 0.8], random_seed=TORCH_SEED):
    """
    Split a dataset (or Subset) into train/validation with stratification.
    Returns quant_data, quant_data_labels, quant_test, quant_test_labels.
    """
    # --- 1. Extract labels for stratification ---
    if isinstance(dataset, data.Subset):
        # dataset.dataset is the original dataset
        base = dataset.dataset
        if hasattr(base, "targets"):
            full_targets = base.targets
        elif hasattr(base, "labels"):
            full_targets = base.labels
        else:
            # Fallback: read label from each item
            full_targets = [base[i][1] for i in range(len(base))]
        # Now pick out only those labels in the Subset
        labels = [full_targets[i] for i in dataset.indices]
    else:
        # Not a Subset: pull directly
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            labels = [dataset[i][1] for i in range(len(dataset))]

    # --- 2. Compute split indices ---
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=split[1],
        random_state=random_seed,
        stratify=labels
    )

    # --- 3. Build Subsets ---
    train_ds = data.Subset(dataset, train_idx)
    val_ds   = data.Subset(dataset, val_idx)

    # --- 4. Batch up train (“quant_data”) ---
    if len(train_ds) > 0:
        qi = train_ds.indices  # shorthand
        # stack inputs
        quant_data = torch.stack([torch.tensor(dataset[i][0]) for i in qi])\
                         .to(torch.device(DEV))
        # stack labels
        quant_data_labels = torch.tensor([dataset[i][1] for i in qi])\
                                .to(torch.device(DEV))
    else:
        quant_data, quant_data_labels = torch.empty(0), torch.empty(0)

    # --- 5. Batch up val (“quant_test”) ---
    if len(val_ds) > 0:
        vi = val_ds.indices
        quant_test = torch.stack([torch.tensor(dataset[i][0]) for i in vi])\
                            .to(torch.device(DEV))
        quant_test_labels = torch.tensor([dataset[i][1] for i in vi])\
                                 .to(torch.device(DEV))
    else:
        quant_test, quant_test_labels = torch.empty(0), torch.empty(0)

    return quant_data, quant_data_labels, quant_test, quant_test_labels