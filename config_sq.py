import argparse


def get_args():
    parser = argparse.ArgumentParser("SNN-OSBC")
    parser.add_argument('--data_dir', type=str, help='path to the dataset')
    parser.add_argument('--out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')

    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', type=str, help='resume from the checkpoint path')

    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('--tau', default=2.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    return args