import os
import time
import argparse
import sys
import datetime

from modelutils import *
from model_library import DVSGestureNetParametric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import amp
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.datasets import n_mnist, dvs128_gesture

def spikeRateLoss(true_rate, false_rate, model_out, label):
    modified_label = torch.zeros_like(label).to(DEV)
    modified_label[label < 0.5] = false_rate
    modified_label[label > 0.5] = true_rate
    return F.mse_loss(model_out, modified_label)

def print_neuron_parameters(model_layers):
    layers = find_layers(model_layers, layers=[neuron.ParametricLIFNode])
    for k in layers:
        w = layers[k].w
        tau = 1.0 / torch.sigmoid(w)
        print(k, "tau:", tau.item())

def main():
    
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    net = DVSGestureNetParametric(channels=128, tau=args.tau)

    print(net)

    net.to(args.device)
    
    T_MAX = 64

    # 初始化数据加载器
    data_path = "./datas/DVS128Gesture"
    train_dataset = dvs128_gesture.DVS128Gesture(
        root=data_path,
        train=True,
        data_type='frame',
        frames_number=args.T,
        split_by='number',
    )
    test_dataset = dvs128_gesture.DVS128Gesture(
        root=data_path,
        train=False,
        data_type='frame',
        frames_number=args.T,
        split_by='number',
    )
            
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                                                num_workers=args.j, pin_memory=True, sampler=None, persistent_workers=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.b, shuffle=True,
                                                num_workers=args.j, pin_memory=True, persistent_workers=True)
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1


    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        raise NotImplementedError(args.opt)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    out_dir = os.path.join(args.out_dir, f'{type(net).__name__}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'
    
    out_dir += '_dvs'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        idx = 0
        print_neuron_parameters(net.conv_fc)
        for img, label in train_data_loader:
            #print(idx, img.shape)
            idx += 1
            optimizer.zero_grad()
            img = img.to(args.device).to(torch.float32)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

            with amp.autocast(device_type=DEV):
                out_fr = 0.
                for t in range(args.T):
                    img_t = img[:, t, :, :, :]
                    out_fr += net(img_t)
                out_fr = out_fr / args.T
                loss = spikeRateLoss(true_rate=0.80, false_rate=0.02, model_out = out_fr, label = label_onehot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        scheduler.step()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device).to(torch.float32)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr = 0.
                for t in range(args.T):
                    img_t = img[:, t, :, :, :]
                    out_fr += net(img_t)
                out_fr = out_fr / args.T
                #loss = F.mse_loss(out_fr, label_onehot)
                loss = spikeRateLoss(true_rate=0.80, false_rate=0.02, model_out = out_fr, label = label_onehot)
                
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        #print_neuron_parameters(net.conv_fc)

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1] # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)


    with torch.no_grad():
        img, label = test_data_loader[0]
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            img_t = img[:, t, :, :, :]
            out_fr += net(img_t)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy",v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy",s_t_array)


if __name__ == '__main__':
    main()