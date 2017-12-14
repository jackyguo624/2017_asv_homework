import torch.nn as nn
import math
from torch import FloatTensor
from torch.autograd import Variable
import torch

__all__ = [
    'cnn'
]

class CNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super(CNN, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim
        self.timedim = kwargs.get('timedim',0)
        self.channel = kwargs.get('channel',1)
        layers = []
        conv0 = nn.Conv2d(self.channel, 128, kernel_size=(3,5), padding=(1, 0), stride=(1, 2))
        conv1 = nn.Conv2d(128,256, kernel_size=(2,3), stride=(1, 2))
        conv2 = nn.Conv2d(256,256,kernel_size=(2,2),stride=(2,2))

        layers += [conv0, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2))]
        layers += [conv1, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))]
        layers += [conv2, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(4*5*256, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.outputdim),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.reshape_cnn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reshape_cnn(self, x):
        bs, ht, wid = x.size()
        return x.view(bs, self.channel, ht, wid)

    def rearrange_cnn(self, v):
        bs, inputdim = v.size()
        inputs = v.view(bs, self.channel * self.timedim, -1)
        rerange_data = tuple(inputs[:, i::self.channel] for i in range(self.channel))
        batch = torch.cat(rerange_data, dim=1).view(bs, self.channel, self.timedim, -1)
        return batch

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def cnn(inputdim, outputdim,**kwargs):
    model = CNN(inputdim,outputdim,**kwargs)
    return model

if __name__ == '__main__':
    a = Variable(FloatTensor(2,3*23*40), requires_grad=True)
    print FloatTensor(2,3*23*40).size()
    cnn = cnn(1920,1920)
    cnn.forward(a)
