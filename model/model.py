from torchvision.models import resnet50
from torch import nn

class ResnetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResnetModel, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
    def update(self, x, y):
        outputs = self(x)
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

