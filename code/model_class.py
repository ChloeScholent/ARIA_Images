import torch
from torch import nn


class PowerliftingCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),  # downsample
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
        )
        # compute output shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            n_features = self.cnn_layers(dummy).numel()
        
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, num_classes)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = self.linear_layers(out)
        return out



class PowerliftingLandmarks(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layer_size = [input_size, 92, 50, 26, 12, num_classes]

        self.layers.append(nn.Flatten())
        for i in range(len(self.layer_size)-2):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.layer_size[-2], self.layer_size[-1]))        

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out