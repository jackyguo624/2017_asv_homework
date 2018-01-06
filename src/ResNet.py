import torch.nn as nn
import math


__all__ = ['ResNet', 'resnet18', 'resnet18_cqcc', 'resnet18_fft']


def conv3_7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 7), stride=stride,
                     padding=(1, 3), bias=False)


def conv3_3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, name, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if name == 'cqcc':
            self.conv1 = conv3_7(inplanes, planes, stride)
            self.conv2 = conv3_7(planes, planes)
        elif name == 'fft':
            self.conv1 = conv3_3(inplanes, planes, stride)
            self.conv2 = conv3_3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3_7(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, name, block, layers, num_class=2):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self._construct_layer(name, block, layers, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _construct_layer(self, name, block, layers, num_class):
        if name == 'cqcc':
            conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=(2, 3),
                              padding=(1, 3), bias=False)
            bn1 = nn.BatchNorm2d(64)
            relu = nn.ReLU(inplace=True)
            maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.conv = nn.Sequential(conv1, bn1, relu, maxpool)

            layer1 = self._make_layer(name, block, 64, layers[0])
            layer2 = self._make_layer(name, block, 128, layers[1], stride=(2, 3))
            layer3 = self._make_layer(name, block, 256, layers[2], stride=(2, 3))

            self.reslayers = nn.Sequential(layer1, layer2, layer3)
            # output = (3, 8)
            self.avgpool = nn.AvgPool2d((1, 3), stride=1)
            # output = (3, 6)
            self.fc = nn.Linear(256 * 3 * 6, num_class)
        elif name == 'fft':
            conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            bn1 = nn.BatchNorm2d(64)
            relu = nn.ReLU(inplace=True)
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv = nn.Sequential(conv1, bn1, relu, maxpool)
            
            layer1 = self._make_layer(name, block, 64, layers[0])
            layer2 = self._make_layer(name, block, 128, layers[1], stride=2)
            layer3 = self._make_layer(name, block, 256, layers[2], stride=2)
            layer4 = self._make_layer(name, block, 512, layers[3], stride=2)

            self.reslayers = nn.Sequential(layer1, layer2, layer3, layer4)
            
            self.avgpool = nn.AvgPool2d(7, stride=1)
            
            self.fc = nn.Linear(512 * 3, num_class)

    def _make_layer(self, name, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(name, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(name, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs, ht, wid = x.size()
        x = x.view(bs, 1, ht, wid)
        x = self.conv(x)

        x = self.reslayers(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18():
    model = ResNet('cqcc', BasicBlock, [2, 2, 2])
    return model


def resnet18_cqcc():
    model = ResNet('cqcc', BasicBlock, [2, 2, 2])
    return model


def resnet18_fft():
    model = ResNet('fft', BasicBlock, [2, 2, 2, 2])
    return model

def test():
    import torch
    x = torch.FloatTensor(1, 257, 200)
    x = torch.autograd.Variable(x, requires_grad=False)
    resnet = resnet18('fft')
    print (resnet(x))


if __name__ == '__main__':
    test()
