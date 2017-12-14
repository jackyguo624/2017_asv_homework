import torch.nn as nn
import torch.nn.functional as F
import math


class DNN(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(DNN, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.hiddim = 4096
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, self.hiddim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hiddim, self.hiddim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hiddim, self.hiddim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hiddim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, self.outputdim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, self.inputdim)
        x = self.classifier(x)
        return x
