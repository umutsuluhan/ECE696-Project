import torch
import snntorch as snn
import torch.nn as nn
import torch.nn.functional as FF

from snntorch import surrogate
from snntorch import utils

beta = 0.9  # neuron decay rate
spike_grad = surrogate.fast_sigmoid(slope=25)  # surrogate gradient

class SNNYOLO(nn.Module):
    def __init__(self, num_classes, num_boxes=7, grid_size=7):
        super(SNNYOLO, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # SNN layer is defined below, beta is the decay rate and spike_grad is the backpropogation technique
        self.snn1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.snn2_1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.snn3_1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.snn4_1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, grid_size * grid_size * num_boxes * (5 + num_classes))
        )

        self.snn6 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def reset_states(self):
        for module in self.modules():
            if hasattr(module, 'mem'):
                module.mem = None

    def forward(self, x, num_steps=10):
        batch_size = x.size(0)

        # Memory for each spiking neuron is initialized
        mem1 = self.snn1.init_leaky()
        mem2_1 = self.snn2_1.init_leaky()
        mem3_1 = self.snn3_1.init_leaky()
        mem4_1 = self.snn4_1.init_leaky()
        mem6 = self.snn6.init_leaky()

        # Spike and memory recording lists
        spk_rec = []
        mem_rec = []

        stacked_img = x 
        # For each step, a forward pass is performed
        for step in range(num_steps):
            x = stacked_img[:, step*3:(step+1)*3, :, :]

            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            # Spike and memory for spiking neuron are recorded 
            spk1, mem1 = self.snn1(x1, mem1)
            x1 = self.relu(spk1)
            x1 = self.maxpool(x1)

            identity = x1
            x2 = self.conv2_1(x1)
            x2 = self.bn2_1(x2)
            spk2_1, mem2_1 = self.snn2_1(x2, mem2_1)
            x1 = self.relu(identity + spk2_1)

            identity = self.downsample3(x1)
            x3 = self.conv3_1(x1)
            x3 = self.bn3_1(x3)
            spk3_1, mem3_1 = self.snn3_1(x3, mem3_1)
            x1 = self.relu(identity + spk3_1)

            identity = self.downsample4(x1)
            x4 = self.conv4_1(x1)
            x4 = self.bn4_1(x4)
            spk4_1, mem4_1 = self.snn4_1(x4, mem4_1)
            x1 = self.relu(identity + spk4_1)

            pooled = self.avgpool(x1)
            flattened = torch.flatten(pooled, 1)
            head_out = self.head(flattened)
            head_out = head_out.view(batch_size, self.grid_size, self.grid_size, self.num_boxes, 5 + self.num_classes)

            spk6, mem6 = self.snn6(head_out, mem6)
            # Last spike and memory encoding are appended to the lists
            spk_rec.append(spk6)
            mem_rec.append(mem6)

        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)
        return spk_rec, mem_rec
    