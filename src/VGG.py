import torch.nn as nn
import math


__all__ = [
    'VGG', 'cnn_fft'
]


class VGG(nn.Module):

    def __init__(self, features, inputdim, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        hidden = 4096
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 48, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        bs, hs, wid = x.size()
        x = x.view(bs, 1, hs, wid)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def cnn_fft(inputdim, num_classes, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG(make_layers(cfg['E'], batch_norm=True),inputdim, num_classes, **kwargs)

    return model


def test():
    import torch
    vgg = cnn_fft(123, 2)
    x = torch.FloatTensor(1, 257, 200)
    x = torch.autograd.Variable(x, requires_grad=True)
    vgg(x)


if __name__ == "__main__":
    test()
