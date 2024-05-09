from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Featurizer(nn.Module):
    def __init__(self, num_classes):
        super(Featurizer, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        del self.resnet.fc
        self.resnet.fc = Identity()  

    def forward(self,x):
        return self.resnet(x)

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, out_features))
    else:
        return nn.Linear(in_features, out_features)

class ResnetBase(nn.Module):
    def __init__(self, num_classes, is_nonlinear=False):
        super(ResnetBase, self).__init__()
        self.featurizer = Featurizer(num_classes)
        self.classifier = Classifier(2048, num_classes, is_nonlinear)
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def forward(self, x):
        return self.network(x)
    
    def update(self, x, y):
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def update_lr(self):
        self.scheduler.step()