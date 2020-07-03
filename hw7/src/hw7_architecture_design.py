"""hw7_Network_Compression (Architecuture Design)"""
# Depthwise & Pointwise Convolution

import torch.nn as nn
import torch.nn.functional as F
import torch

class StudentNet(nn.Module):
    """
    use Depthwise & Pointwise Convolution Layer to construct model
    """
    def __init__(self, base=16, width_mult=1):
        """
        Args:
        base: initial input channel number, *2 after every layer, until *16
        width_mult: Network Pruning, in base*8 chs' Layer use * width_mult to represent pruned ch number        
        """
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: every Layers' ch number
        bandwidth = [ base * m for m in multiplier]

        # only perform Pruning after the 3rd Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # don't break down the first Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 confines Neuron's min = 0，max = 6
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # after Pointwise Convolution, no need to go through ReLU，Pointwise + ReLU's effect is bad
                nn.MaxPool2d(2, 2, 0),
                # Down Sampling at the end of block
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # from this block, have Down Sampled many times，so don't perform MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # turn different input image into the same size
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)