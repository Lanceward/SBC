# In this script, I will try to apply Optimal Brain Surgeon for Spiking Neural Network
# I will be doing per layer Pruning, i.e. quantizing weights to 0
# Proofs of algorithm can be seen in the two pdfs I put in here

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import spikingjelly.activation_based.layer as sj_layer
from spikingjelly import activation_based
from spikingjelly.datasets import n_mnist, transforms, dvs128_gesture, split_to_train_test_set
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from modelutils import *

# if 'cuda' in DEV:
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False

REPORT_TIME = True
TIMING_INTERVAL_S = 0.5

MAX_MATRIX_SIZE = 2**29
MAX_NUM_ROW_BATCH = 2**8

MBP = False # magnitude based pruning

def LAMPS(model_layers, power = 2):
    linear_layers = find_layers(model_layers)
    
    layers_linearized = {}
    lamps_score = {}
    for name, layer in linear_layers.items():
        w = layer.weight.clone().flatten()
        w2 = w.abs().pow(power)
        # w2 = torch.square(w)
        w2_fliped, _ = torch.sort(w2, descending=True) # sort in descending order
        w2_cumsum = torch.flip(torch.cumsum(w2_fliped, dim=0), dims=(0,))
        w2 = torch.flip(w2_fliped, dims=(0,))
        lamps_score[name] = w2/w2_cumsum

    return lamps_score

def get_prune_percs_from_lamps(lamps_score, target_percentages):
    big_list_of_lamps_scores = []
    for name, lamps in lamps_score.items():
        big_list_of_lamps_scores.append(lamps)
    
    big_list_of_lamps_scores = torch.concat(big_list_of_lamps_scores)
    big_list_of_lamps_scores, _ = torch.sort(big_list_of_lamps_scores)
    
    target_lamp_scores = []
    for target_perc in target_percentages:
        target_num_pruned = min(int(big_list_of_lamps_scores.shape[0] * target_perc), big_list_of_lamps_scores.shape[0]-1)
        target_lamp_scores.append(big_list_of_lamps_scores[target_num_pruned])
        
    prune_perc = {}
    for name, lamps in lamps_score.items():
        num_to_prune_at_each_target = []
        for target_lamp_score in target_lamp_scores:
            num_to_prune_this_layer = torch.searchsorted(lamps, target_lamp_score)
            num_to_prune_at_each_target.append((num_to_prune_this_layer / lamps.shape[0]).item())
        prune_perc[name] = num_to_prune_at_each_target
        
    return prune_perc

def get_prune_percs_from_lamps_single_target(lamps_score, target_perc):
    # 1) collect layer names & lengths
    layer_names = list(lamps_score.keys())
    lengths     = [lamps_score[name].numel() for name in layer_names]

    # 2) build offsets so we know which slice of the big tensor is which layer
    offsets = [0]
    for L in lengths:
        offsets.append(offsets[-1] + L)

    # 3) concat all scores into one flat vector
    big_scores = torch.cat([lamps_score[name].view(-1) for name in layer_names])

    # 4) how many in total to prune?
    total = big_scores.numel()
    target_num_pruned = min(int(total * target_perc), total - 1)

    # 5) global sort + take smallest K
    sorted_vals, sorted_idx = torch.sort(big_scores, stable=True)
    topk_idx = sorted_idx[:target_num_pruned]

    # 6) for each layer, count how many topk_idx fall into its slice
    prune_perc = {}
    for i, name in enumerate(layer_names):
        start = offsets[i]
        end   = offsets[i+1]
        # boolean mask of which global indices are in [start,end)
        in_layer = (topk_idx >= start) & (topk_idx < end)
        num_prune = int(in_layer.sum().item())
        prune_perc[name] = num_prune / lengths[i]

    return prune_perc


class SurrogateVoltage:
    #Implementation assumes each pass contains all data
    def __init__(self, nname, llayer, data_length, tau, T, spacial_size = [], is_osbs = True, batchnormlayer = None):
        self.name = str(nname)
        self.layer = llayer
        self.dev = self.layer.weight.device
        self.tau = tau
        self.T = T
        self.data_count = 0
        self.data_length = data_length
        self.linearized_data_count = 0
        self.is_osbs = is_osbs
        if batchnormlayer is not None:
            self.hasbatchnorm = True
            bn = batchnormlayer
            # Given a BatchNorm2d layer `bn` and input X of shape [C, H, W]:
            gamma    = bn.weight.data                    # [C]
            var   = bn.running_var                     # [C]
            eps   = bn.eps                             # scalar
            # Compute per-channel constants
            self.AFF = gamma / torch.sqrt(var + eps)      # [C]
            #print("BatchNorm Para", self.AFF, self.AFF.shape, gamma, gamma.shape)
        else:
            self.hasbatchnorm = False
        
        if isinstance(self.layer, sj_layer.Linear):
            #parameters
            self.linearized_in_features = self.layer.in_features
            self.linearized_out_features = self.layer.out_features
            
            #init weight
            self.W = self.layer.weight.clone()
            # print("Linear layer parameters: ", self.W.shape)        
        elif isinstance(self.layer, sj_layer.Conv2d):
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

            #init weight
            self.W = self.layer.weight.clone().flatten(1) #(out_channel, in_channel, kernel_size0, kernel_size1) -> (out_channel, in_channel * kernel_size0 * kernel_size1)
        else:
            raise ValueError(type(self.layer) + " unsupported")

        print("    layer summary: tau:", self.tau, "T", self.T, "Linearized parameters:", self.linearized_in_features, self.linearized_out_features)

        self.H = torch.zeros((self.linearized_in_features, self.linearized_in_features)).to(self.dev) # Hessian
        self.M = torch.zeros_like(self.W).to(torch.bool).to(self.dev) # prune mask

        #build convolution matrix
        if self.tau > 0.0:
            self.conv = torch.zeros((T, T)).to(self.dev)
            conv_row = torch.zeros((1, T)).to(self.dev)
            for t in range(T):
                conv_row[0, t] = ((self.tau - 1)/self.tau)**(T-t-1)
            for i in range(T):
                self.conv[i, :i+1] = conv_row[0, T-i-1:]
        else:
            # self.conv = torch.ones((T, T)).to(self.dev)
            self.conv = torch.eye(T).to(self.dev)
    
        # print("Conv: ", self.conv)
    
        if MBP:
            weight_flatten = self.W.flatten().clone()
            weight_flatten = torch.square(weight_flatten)
            self.weight_mag_asc = torch.argsort(weight_flatten)
            self.mbp_idx = 0
    
    # Add B sample worth of data. 
    # Assume input has size (T, B, I)
    # Assume output has size (T, B, O)
    def add_data_sample(self, inp, out):
        #assert inp[0].shape[1] == 1 and out.shape[1] == 1 # assert theres only 1 input
        inp = inp[0]
        B = inp.shape[1]
        if isinstance(self.layer, sj_layer.Linear):
            m = B
            #inp = inp[:, 0, :] #(T, 1, in) -> (T, in_features)
            inp = torch.transpose(inp, dim0=0, dim1=1) #(B, T, I)
            if self.is_osbs:
                KS = torch.matmul(self.conv, inp) #(T, T) x (B, T, I) -> (B, T, I)
            else:
                KS = inp # (B, T, I)
            KS = KS.flatten(0, 1) #(B*T, I)
        elif isinstance(self.layer, sj_layer.Conv2d):
            inp = torch.transpose(inp, dim0=0, dim1=1) #(B, T, C, H, W)
            if self.is_osbs:
                inp = torch.einsum('ti,bichw->btchw', self.conv, inp) #(T, T) x (B, T, C, H, W) -> (B, T, C, H, W)
            inp = inp.flatten(0, 1) #(B, T, C, H, W) -> (B*T, C, H, W)
            inp = self.unfold(inp) #(B*T, C, H, W) -> (B*T, C * kernel0 * kernel1, L)
            m = inp.shape[2] * B
            inp = inp.permute([1, 0, 2]) #(B*T, C * kernel0 * kernel1, L) -> (C * kernel0 * kernel1, B*T, L)
            KS = inp.flatten(1) #(C * kernel0 * kernel1, B*T, L) -> (C * kernel0 * kernel1, B*T*L)
            KS = KS.t() # (C * kernel0 * kernel1, B*T*L) -> (B*T*L, C * kernel0 * kernel1)
        else:
            raise ValueError(type(self.layer) + " unsupported")
        
        self.data_count += B
        
        batch_H = 2 * KS.t().matmul(KS) / m
        self.linearized_data_count += m
        delta = batch_H - self.H
        self.H += (delta * m) / self.linearized_data_count
        
        print("\rlayer hessian progress: " + "{:.2f}".format(self.data_count/self.data_length * 100) + "%", end='')

        
    def calculate_Hessian(self, layer_prune_target, col_batch: int = 2, row_batch: int = 1):
        # assume all data samples have been added
        assert self.data_count == self.data_length and self.H.shape[0] == self.linearized_in_features
        print("Layer " + self.name + " data length: " + str(self.data_count) + ", linearized data length: " + str(self.linearized_data_count))

        H = self.H.clone()
        # No need for different Hessians for each neuron during pruning
        percdamp = 0.01
        damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.linearized_in_features, device=self.dev)
        # H[diag, diag] += damp
        # if DEV == 'mps': # mps do not support gpu inverse
        #     self.H_inv = torch.inverse(H.to('cpu')).to(DEV)
        # else:
        #     self.H_inv = torch.inverse(H)
        # del H
        # # self.H_inv and self.H will not be modified for the rest of calculate_Hessian()

        # !! FAST & MEMORY INTENSIVE !!
        # Calculate the pruning order of weights per channel
        print("Layer " + self.name + " calculate weight order...")
        clock_ordering = time.time()
        # I want a 2d tensor of the same shape as self.W, but recording the loss of each weight
        losses_W = torch.zeros_like(self.W).to(DEV)
        for start_column in range(0, self.linearized_out_features, col_batch):
            torch_empty_cache()
            # get current batch size
            current_batchsize = min(col_batch, self.linearized_out_features-start_column)
            
            # expand H into current_batchsize counts
            batchH = self.H.unsqueeze(0).repeat(current_batchsize, 1, 1)

            # add batchnorm
            if self.hasbatchnorm and self.is_osbs:
                batchH = batchH * (self.AFF[start_column: start_column+current_batchsize].view(-1, 1, 1) 
                                   * self.AFF[start_column: start_column+current_batchsize].view(-1, 1, 1) )

            # for numerical stability
            damp = percdamp * torch.mean(torch.diagonal(batchH, dim1=1, dim2=2), dim=1)
            diag = torch.arange(self.linearized_in_features, device=self.dev)
            
            batchH[:, diag, diag] += damp.unsqueeze(1).repeat(1, self.linearized_in_features)
            
            # calculate inverse hessian
            channels = []
            batchH_invs = batchH#torch.zeros_like(batchH)
            for j in range(current_batchsize):
                if damp[j] < 1e-7: # all diagonals are 0
                    # prune all weights pointing to this channel
                    self.W[start_column+j, :] = 0
                    self.M[start_column+j, :] = True
                    batchH_invs[j] = torch.eye(self.linearized_in_features, device=DEV) * float('inf')
                else:
                    if DEV == 'mps': # mps do not support gpu inverse in large matrix
                        batchH_invs[j] = torch.inverse(batchH[j].to('cpu')).to(DEV)
                    else:
                        batchH_invs[j] = torch.inverse(batchH[j])
                    channels.append(j)
            channels = torch.tensor(channels, device=self.dev, dtype=torch.int)
            
            # pull out active channels for pruning
            active_channels_global_idx = start_column+channels

            temp_Ws = self.W[active_channels_global_idx].clone()
            temp_Masks = self.M[active_channels_global_idx].clone().bool()
            temp_Hinvs = batchH_invs[channels] # batchH is already in local index
            
            # matrix initialization
            losses = torch.empty_like(temp_Ws)
            Hinv_diags = torch.empty((channels.numel(), self.linearized_in_features), dtype=torch.float32, device=DEV)
            temp_range = torch.arange(channels.numel(), device=DEV)

            # because each channel has a seperate hessian that does not
            # interfere with each other, we can do the ordering of 
            # all the channels in parallel
            time1 = 0
            speed_timer = 0
            ordered_weight_count = 0
            for widx in range(0, self.linearized_in_features, row_batch):
                if widx != 0 and widx % (row_batch*10) == 0:
                    now_ordered_weight_count = start_column*self.linearized_in_features + widx*current_batchsize
                    ordering_speed = (self.W.numel()-now_ordered_weight_count)/(now_ordered_weight_count - ordered_weight_count) * (time.time() - speed_timer)
                    print("\rweight ordering progress: " + "{:.2f}".format(now_ordered_weight_count/self.W.data.numel() * 100) + "% "\
                        + "{:.2f}".format(ordering_speed) + "s " + str(time1//1000), end='          ')
                    ordered_weight_count = now_ordered_weight_count
                    speed_timer = time.time()

                time1 = time.time_ns()

                # get losses for each unpruned weights
                Hinv_diags.copy_(torch.diagonal(temp_Hinvs, dim1=1, dim2=2))
                losses.copy_(temp_Ws).pow_(2).div_(Hinv_diags)
                losses.masked_fill_(temp_Masks, float('inf'))
                
                # pick the per-channel top k smallest
                current_row_batch = min(row_batch, self.linearized_in_features-widx)
                _, ps = torch.topk(losses, current_row_batch, dim=1, largest=False) #[C, current_row_batch]
                
                # register the pruned weights and their losses
                range_batch = temp_range.unsqueeze(1).expand((-1, current_row_batch)) #[C] -> [C, current_row_batch]
                active_channels_global_batch = active_channels_global_idx.unsqueeze(1).expand((-1, current_row_batch)) #[C] -> [C, current_row_batch]
                
                temp_Masks[range_batch, ps] = True
                losses_W[active_channels_global_batch, ps] = losses[range_batch, ps]
                
                #prune & compensate weights
                wps = temp_Ws[range_batch, ps] #[C, current_row_batch]
                U = temp_Hinvs[range_batch, :, ps].transpose(dim0=1, dim1=2) #[C, F, current_row_batch]
                Ut = temp_Hinvs[range_batch, ps, :] # !! important! for floating point error, U will drift from U^T slightly
                Hinv_J = temp_Hinvs[range_batch, ps, :][range_batch, :, ps]
                HJ_inv = torch.inverse(Hinv_J)

                delta = torch.bmm(U, torch.bmm(HJ_inv, wps.unsqueeze(2)))
                temp_Ws -= delta.squeeze(2)
                
                # Hessian-inverse update
                temp_Hinvs.baddbmm_(U, torch.bmm(HJ_inv, Ut), beta=1.0, alpha=-1.0)

                time1 = time.time_ns() - time1
        
        print("finished. Took " + "{:.2f}".format((time.time() - clock_ordering)) + "s. Average " + "{:.2f}".format((time.time() - clock_ordering) / self.linearized_in_features * 1000) + "ms/column")

        # find weight pruning order
        self.weight_pruning_order = torch.zeros_like(self.W, dtype=torch.int32).to(DEV)
        self.weight_pruning_order = torch.argsort(losses_W, dim=1)
        
        #find number of weights to prune from each neuron
        n_rows, n_cols = losses_W.shape
        flat = losses_W.reshape(-1)
        # indices of the N smallest entries, no full sort needed
        _, flat_idx = torch.topk(flat, k=layer_prune_target, largest=False, sorted=False)
        # map flat indices -> row numbers
        row_idx = flat_idx // n_cols                      # integer division
        # how many times does each row occur?
        self.perchannel_weight_pruning_count = torch.bincount(row_idx, minlength=n_rows).to(dtype=torch.int32, device=DEV)     # k.shape == (n_rows,)
        # print(self.perchannel_weight_pruning_count)

        # initialize current pruning progress
        self.perchannel_weight_pruning_progress = torch.sum(self.M, dim=1).to(dtype=torch.int32, device=DEV)
        # print(self.perchannel_weight_pruning_progress)
    
    @torch.no_grad()
    def fastprune(self, current_pruned, prune_target, col_batch: int = 2, row_batch: int = 1):
        if current_pruned >= prune_target:
            return
        I = self.linearized_in_features

        H = self.H.clone()
        percdamp = 0.01
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.linearized_in_features, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        
        last_time = time.time()
        last_index = current_pruned
        # FAST pruning, prune col_batch neurons together
        for start_column in range(0, self.linearized_out_features, col_batch):
            # get current batch size
            B = min(col_batch, self.linearized_out_features-start_column)

            batch_weight_pruning_progress = self.perchannel_weight_pruning_progress[start_column:start_column+B]
            batch_weight_pruning_count = self.perchannel_weight_pruning_count[start_column:start_column+B]
            batch_weight_pruning_order = self.weight_pruning_order[start_column:start_column+B]

            # construct permuter and unpermuter
            inverse_wp_order = batch_weight_pruning_order.clone().flip(1)
            inverse_wp_order_inv = torch.argsort(inverse_wp_order)
            # expand Hinv into current_batchsize counts
            # permute Hessian depending on pruning order
            inverse_wp_order_row = inverse_wp_order.unsqueeze(-1).expand(-1, -1, self.linearized_in_features)
            inverse_wp_order_col = inverse_wp_order.unsqueeze(1).expand(-1, self.linearized_in_features, -1)
            Hinv_batch_ordered = Hinv.unsqueeze(0).expand(B, I, I).clone()
            Hinv_batch_ordered = Hinv_batch_ordered.gather(1, inverse_wp_order_row)
            Hinv_batch_ordered = Hinv_batch_ordered.gather(2, inverse_wp_order_col)
            
            batch_W = self.W[start_column:start_column+B]
            batch_M = self.M[start_column:start_column+B]
            
            batch_W_ordered = batch_W.gather(1, inverse_wp_order) #order the weights such that the earlier the prune, further back it is
            
            B_idx = torch.arange(B).to(self.dev) #[B]
            
            # timings
            losses = 0
            i = 0
            while i < self.linearized_in_features:                
                # next to prune index for each row
                active_channels = (batch_weight_pruning_progress < batch_weight_pruning_count) #[B]
                if not active_channels.any():# or i > 15:
                    break
                # calcualte current row batch (RB) size.
                RB = min(row_batch, (batch_weight_pruning_count-batch_weight_pruning_progress)[active_channels].min().item())
                ordered_i_end = self.linearized_in_features-i
                ordered_i = ordered_i_end - RB
                input_idxes = batch_weight_pruning_order[:, i:i+RB] #[B]
                B_idx_batch = B_idx.unsqueeze(-1).expand(-1, RB)
                
                # step 1 grab the weight and mask
                wps = batch_W_ordered[:, ordered_i:ordered_i_end].clone() # [B, RB]
                batch_M[B_idx_batch[active_channels], input_idxes[active_channels]] = True

                # step 2 weight compensation and Hessian fix
                Hinv_batch_remaining = Hinv_batch_ordered[:, :ordered_i_end, :ordered_i_end]# [B, R, R]
                HinvJ = Hinv_batch_remaining[:, ordered_i:, :] # last RB active columns [B, RB, R]
                HinvJT = Hinv_batch_remaining[:, :, ordered_i:] # last RB active rows [B, RB, R]
                EHET = Hinv_batch_remaining[:, ordered_i:, ordered_i:] #ej @ H^-1 @ ej^T [B, RB, RB]
                EHET_inv = torch.inverse(EHET) # [B, RB, RB], watchout for inverse errors
                delta = torch.bmm(HinvJT, torch.bmm(EHET_inv, wps.unsqueeze(-1))) # [B, R, 1]
                batch_W_ordered[active_channels, :ordered_i_end] -= delta[active_channels].squeeze(2)
                Hinv_batch_remaining.baddbmm_(HinvJT, torch.bmm(EHET_inv, HinvJ), beta=1.0, alpha=-1.0)

                batch_weight_pruning_progress[active_channels] += RB
                current_pruned += torch.sum(active_channels).item() * RB
                i += RB

                torch_synchronize()
                # reporting per Time Interval
                if REPORT_TIME and time.time() - last_time >= TIMING_INTERVAL_S:
                    losses = (wps**2 * EHET_inv.diagonal(dim1=1, dim2=2))[active_channels]
                    now_time = time.time()
                    pruning_speed = (current_pruned - last_index) / (now_time - last_time)
                    W_unordered = self.W.clone() # I just copied current pruning progress into clone of W
                    W_unordered[start_column:start_column+B] = batch_W_ordered
                    print("\r" + format(current_pruned/prune_target, "05.2%") \
                        + " idx: " + str(current_pruned) \
                        + ", pspeed: " + "{:.2f}".format(pruning_speed) \
                        + "/s, remaining: " + str(round((prune_target-current_pruned)/(pruning_speed+1))) \
                        + "s, loss: " + str(format(torch.mean(losses).item(), "0.4g")) \
                        + " CHECK: " + str(torch.sum(self.M).item()) + "|" + str(torch.sum(torch.abs(W_unordered) < 5e-7).item()), end='')
                    last_index = current_pruned
                    last_time = now_time
            batch_W_reordered = batch_W_ordered.gather(1, inverse_wp_order_inv) # reorder the pruned and compensated batchW
            self.W[start_column:start_column+B] = batch_W_reordered
            # print()
        print() # next line for the progress bar

    def fastpruneMBP(self, current_pruned, prune_target):
        if current_pruned >= prune_target:
            return
        print("Current pruned: ", current_pruned)
        for _ in range(prune_target - current_pruned):
            self.pruneMBP()

    def pruneMBP(self):
        # quick and easy magnitude based pruning method
        #get flattened index of next weight to prune
        flatten_idx = self.weight_mag_asc[self.mbp_idx]
        row_idx = (flatten_idx // self.linearized_in_features).item()
        col_idx = (flatten_idx % self.linearized_in_features).item()
        loss = self.W[row_idx, col_idx].pow(2)
        self.W[row_idx, col_idx] = 0
        self.M[row_idx, col_idx] = True
        
        self.mbp_idx += 1
        return loss

    def getWeightCopy(self):
        if isinstance(self.layer, sj_layer.Linear):
            w = self.W.clone()
            w[self.M] = 0
            return w
        elif isinstance(self.layer, sj_layer.Conv2d):
            flattened_w = self.W.clone()
            flattened_w[self.M] = 0
            w = flattened_w.unflatten(1, (self.layer.in_channels, self.layer.kernel_size[0], self.layer.kernel_size[1]))
            return w
        else:
            raise ValueError(type(self.layer) + "Not supported")

# define hook
def record_io(sv):
    def tmp(_, inp, out):
        sv.add_data_sample(inp, out)
    return tmp

def optimal_spiking_brain_surgeon(model, input_sample_dataset, tau: dict, T, prune_perc: dict, model_output_dir, OSBS, validation_batch_size, real_model = None):
    for layer_name, perc in prune_perc.items():
        assert perc >= 0.0 and perc <= 1.0

    if real_model is None:
        real_model = model

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
    # assert all layers have pruning percentage
    for name, layer in full_set.items():
        assert prune_perc[name] != None

    svs = {}
    handles = []
    for name, layer in full_set.items():
        print("Name:", name, layer, end='')
        if name+1 in batchnorm_layers:
            print(" has batchnorm")
            # if the following layer is batchnorm
            sv = SurrogateVoltage(name, layer, sample_dataset_shape[0], tau[name], T, spacial_size=[sample_dataset_shape[3], sample_dataset_shape[4]], is_osbs=OSBS, batchnormlayer = batchnorm_layers[name+1])
        else:
            print()
            # create surrogate voltage calculator
            sv = SurrogateVoltage(name, layer, sample_dataset_shape[0], tau[name], T, spacial_size=[sample_dataset_shape[3], sample_dataset_shape[4]], is_osbs=OSBS)
        handles.append(layer.register_forward_hook(record_io(sv)))
        svs[name] = sv

    if not MBP:
        functional.set_step_mode(real_model, 'm')

        #calculate new inputs for layer to be pruned
        for input_sample, _ in sample_dataset_loader: # [B, T, P, H, W]
            functional.reset_net(real_model)
            input_sample = input_sample.transpose(0, 1).to(DEV) # [T, B, P, H, W]
            real_model(input_sample)
        functional.reset_net(real_model)

    #remove hook and unnecessary data
    for h in handles:
        h.remove()
    torch_empty_cache()

    # for each modules, calculate Hessian then prune to target
    for name, sv in svs.items():
        # Initialize weight tracking variables
        layer_weight_count = sv.layer.weight.numel()
        final_prune_count = int(np.floor(layer_weight_count * prune_perc[name]))
        
        # find appropriate batch size such that
        # 1. biggest matrix <= MAX_MATRIX_SIZE
        # 2. number of python row loops in each batch <= MAX_NUM_ROW_BATCH
        in_feature = sv.linearized_in_features
        out_feature = sv.linearized_out_features
        col_batch_size = min(out_feature, math.floor(MAX_MATRIX_SIZE / in_feature**2))
        if isinstance(sv.layer, sj_layer.Linear):
            row_num_of_batches = min(in_feature, MAX_NUM_ROW_BATCH)
            row_batch_size_H = math.ceil(in_feature / row_num_of_batches)
        elif isinstance(sv.layer, sj_layer.Conv2d):
            # row_batch_size_H = 1
            row_num_of_batches = min(in_feature, MAX_NUM_ROW_BATCH)
            row_batch_size_H = math.ceil(in_feature / row_num_of_batches)
        else:
            raise ValueError(type(sv.layer) + " unsupported")
        print("\nLayer", name, "to be pruned. Column batch size", col_batch_size, "row batch:", row_batch_size_H)
        
        if MBP:
            current_prune_count = int(np.floor(torch.sum(sv.M).item()))
            sv.fastpruneMBP(current_prune_count, final_prune_count)
        else:
            # Calculate Hessian
            sv.calculate_Hessian(final_prune_count, col_batch=col_batch_size, row_batch=row_batch_size_H)
            torch_empty_cache()
            
            # Prune
            current_prune_count = int(np.floor(torch.sum(sv.M).item()))
            sv.fastprune(current_prune_count, final_prune_count, col_batch=col_batch_size, row_batch=row_batch_size_H)
        torch_empty_cache()

    return {}, svs

def validate_model(model, T, validation_dataset, pruning_perc = 0.0, out_dir = "", batch = 32):
    if isinstance(pruning_perc, float):
        model_prefix = "{:.3f}".format(pruning_perc)
    elif isinstance(pruning_perc, str):
        model_prefix = pruning_perc
    model_name = model_prefix + "_pruned_model.pth"
    functional.set_step_mode(model, 'm')
    total_correct = 0

    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch, shuffle=True,
                                                num_workers=8, pin_memory=True, sampler=None, 
                                                persistent_workers=False, prefetch_factor=4)

    for validation_data, label in tqdm(validation_data_loader):
        #Return model accuracy on inputs and labels
        validation_data = validation_data.to(DEV)
        validation_data = validation_data.transpose(0, 1) # necessary?
        label = label.to(DEV)
        # outs = 0.
        # for t in range(T):
        #     outs += model(validation_data[t])
        # outs /= T
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
    
    if len(quant_data) != 0:
        # batch quant_data
        quant_data_indices = quant_data.indices
        quant_data_samples = [dataset[i][0] for i in quant_data_indices]
        quant_data = torch.stack(quant_data_samples).to(torch.device(DEV))
        quant_data_labels = [dataset[i][1] for i in quant_data_indices]
        quant_data_labels = [dataset[i][1] for i in quant_data_indices]
        quant_data_labels = torch.tensor(quant_data_labels).to(torch.device(DEV))
        torch.manual_seed(TORCH_SEED)
        if len(quant_data.shape) == 3: # gray scale photo
            quant_data_batch = torch.zeros_like(quant_data).unsqueeze(1).repeat(1, T, 1, 1)
        else:
            quant_data_batch = torch.zeros_like(quant_data).unsqueeze(1).repeat(1, T, 1, 1, 1)

        for t in range(T):
            quant_data_batch[:, t] = encoder(quant_data)
    else:
        quant_data_batch, quant_data_labels = torch.empty(0), torch.empty(0)
    
    
    if len(quant_test) != 0:
        # batch test data
        quant_test_indices = quant_test.indices
        quant_test = [dataset[i][0] for i in quant_test_indices]
        quant_test = torch.stack(quant_test).to(torch.device(DEV))
        quant_test_labels = [dataset[i][1] for i in quant_test_indices]
        quant_test_labels = torch.tensor(quant_test_labels).to(torch.device(DEV))
        torch.manual_seed(TORCH_SEED)
        if len(quant_test.shape) == 3: # gray scale photo
            quant_test_batch = torch.zeros_like(quant_test).unsqueeze(1).repeat(1, T, 1, 1)
        else:
            quant_test_batch = torch.zeros_like(quant_test).unsqueeze(1).repeat(1, T, 1, 1, 1)
            
        for t in range(T):
            quant_test_batch[:, t] = encoder(quant_test)
    else:
        quant_test_batch, quant_test_labels = torch.empty(0), torch.empty(0)
        
    return quant_data_batch, quant_data_labels, quant_test_batch, quant_test_labels

# (B, T, C, W, H)
def split_neuromorphic_data(dataset, split = [0.2, 0.8]):
    # split dataset [0.2, 0.8]. This gives us 1000 sample, 8000 test
    torch.manual_seed(TORCH_SEED)
    quant_data, quant_test = data.random_split(dataset, split)
    
    if len(quant_data) != 0:
        # batch quant_data
        quant_data_indices = quant_data.indices
        quant_data = np.stack([dataset[i][0] for i in quant_data_indices])
        quant_data = torch.tensor(quant_data, device=DEV)
        quant_data_labels = torch.tensor([dataset[i][1] for i in quant_data_indices]).to(DEV)
    else:
        quant_data, quant_data_labels = torch.empty(0), torch.empty(0)

    if len(quant_test) != 0:
        # batch test data
        quant_test_indices = quant_test.indices
        quant_test = np.stack([dataset[i][0] for i in quant_test_indices])
        quant_test = torch.tensor(quant_test, device=DEV)
        quant_test_labels = torch.tensor([dataset[i][1] for i in quant_test_indices]).to(DEV)
    else:
        quant_test, quant_test_labels = torch.empty(0), torch.empty(0)

    return quant_data, quant_data_labels, quant_test, quant_test_labels

# (B, T, C, W, H)
def split_neuromorphic_data_subsets(dataset, split = [0.2, 0.8]):
    # split dataset [0.2, 0.8]. This gives us 1000 sample, 8000 test
    gen = torch.Generator().manual_seed(TORCH_SEED)
    quant_data, quant_test = data.random_split(dataset, split, generator=gen)
    
    return quant_data, quant_test

def stratified_split(dataset, frac=0.8, random_state=TORCH_SEED):
    labels  = [s[1] for s in dataset.samples]  # or dataset.targets
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-frac, random_state=random_state)
    train_idx, val_idx = next(splitter.split(dataset.samples, labels))
    
    train_ds = data.Subset(dataset, train_idx)
    val_ds   = data.Subset(dataset, val_idx)

    return train_ds, val_ds