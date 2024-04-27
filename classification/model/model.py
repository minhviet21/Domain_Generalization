from torchvision.models import resnet50
from torch import nn

class ResnetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResnetModel, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.resnet(x)