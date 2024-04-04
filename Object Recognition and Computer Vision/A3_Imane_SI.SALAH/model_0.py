import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        )

        self.classifier= nn.Sequential(
            # nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=nclasses, bias=True)
        )
    
    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True
    

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad= False
            
    def forward(self, x):
        x= self.features(x)
        x=self.classifier(x)
        return x
