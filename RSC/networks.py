import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from hyperparameter import Hyperparameter
hp = Hyperparameter()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet(torch.nn.Module):
    def __init__(self, hp):
        super(ResNet, self).__init__()
        if hp.model == "ResNet18":
            self.network = torchvision.models.resnet18(weights=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights=True)
            self.n_outputs = 2048

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hp = hp
        self.dropout = nn.Dropout(hp.resnet_dropout)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)