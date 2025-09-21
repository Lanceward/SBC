from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import torch
import torch.nn as nn

class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class DSNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 128, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            
            layer.Linear(128, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class NSNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(34 * 34 * 2, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class GSNNL(nn.Module):
    def __init__(self, tau, v_thres, dropout_rate = 0.2):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1600, 450, bias=True),
            neuron.ParametricLIFNode(init_tau=tau, decay_input=False, v_threshold=v_thres, surrogate_function=surrogate.Sigmoid(alpha=3)),
            
            layer.Dropout(p=dropout_rate),
            
            layer.Linear(450, 10, bias=True),
            neuron.ParametricLIFNode(init_tau=tau, decay_input=False, v_threshold=v_thres, surrogate_function=surrogate.Sigmoid(alpha=3)),
            )
    
    def forward(self, x: torch.Tensor):
        return self.layer(x)

class SNN2L(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            
            layer.Linear(34 * 34 * 2, 256, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            
            layer.Linear(256, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
    
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, tau=2.0):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(neuron.LIFNode(tau=tau, decay_input=False, surrogate_function=surrogate.ATan()))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 500),
            neuron.LIFNode(tau=tau, decay_input=False, surrogate_function=surrogate.ATan()),

            layer.Dropout(0.5),
            layer.Linear(500, 110),
            neuron.LIFNode(tau=tau, decay_input=False, surrogate_function=surrogate.ATan()),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
class DVSGestureNetParametric(nn.Module):
    def __init__(self, channels=128, tau=2.0, decayinput = True):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan()))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 500),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan()),

            layer.Dropout(0.5),
            layer.Linear(500, 110),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan()),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)