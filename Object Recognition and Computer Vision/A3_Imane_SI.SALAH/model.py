import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Load pre-trained ResNet-50
        resnet50 = models.resnet50(pretrained=True)
          
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet50.children())[:-1])

        # Add a new fully connected layer 
        nfeatures= resnet50.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(nfeatures, 512), 
            nn.ReLU(inplace=True),                     
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(512, nclasses)
        )
    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True
    

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad= False
        
    def unfreeze_c1(self):
        for param in self.features.conv1.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x