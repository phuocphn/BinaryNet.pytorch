import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

__all__ = ['alexnet_binary']


class QBinaryConnectNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QBinaryConnectNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=2,bias=False),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=2,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=2,bias=False),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=2,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            

            BinarizeLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),

            nn.Linear(1024, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax()
        )

        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x


# def alexnet_binary(**kwargs):
#     num_classes = kwargs.get( 'num_classes', 1000)
#     return AlexNetOWT_BN(num_classes)
