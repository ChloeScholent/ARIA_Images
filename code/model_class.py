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
    def __init__(self, num_classes):
        super().__init__()
        self.landmarks_layers = nn.Sequential(
            nn.Linear(), 
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.landmarks_layers(x)
        return out
