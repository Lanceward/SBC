from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import torch
import torch.nn as nn

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

class DVSCIFAR10NET_DOWNSIZED(nn.Module):
    def __init__(self, channels=128, tau=2.0, decayinput = True):
        super().__init__()

        conv = []
        for i in range(4):
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
            layer.Linear(500, 100),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan()),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVSCIFAR10NET_FULLSIZED(nn.Module):
    def __init__(self, channels=128, tau=2.0, decayinput = True, detach_reset=False):
        super().__init__()

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),

            layer.Dropout(0.5),
            layer.Linear(512, 100),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),

            layer.VotingLayer(10)
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Input shape: 180, 240
# 180 90 45 22 11 5
# 240 120 60 30 15 8
class ASLDVS_CSNN(nn.Module):
    def __init__(self, tau=2.0, decayinput = True, detach_reset=False, class_num=24):
        super().__init__()
        
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, 12, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(12),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),
            
            layer.Conv2d(12, 32, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),
            
            layer.Conv2d(32, 48, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(48),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(28 * 20 * 48, 24),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Input shape: 180, 240
# 180 90 45 22 11 5
# 240 120 60 30 15 8
class ASLDVS_4CONV_SNN(nn.Module):
    def __init__(self, tau=2.0, decayinput = True, detach_reset=False, class_num=24):
        super().__init__()
        
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, 12, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(12),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),
            
            layer.Conv2d(12, 32, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),
            
            layer.Conv2d(32, 48, kernel_size=5, padding=1, bias=False),
            layer.BatchNorm2d(48),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(48),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(14 * 10 * 48, 24),
            neuron.LIFNode(tau=tau, decay_input=decayinput, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
