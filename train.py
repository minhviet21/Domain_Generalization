from torch import nn
import torch.optim as optim
from model.model import ResnetModel
from dataset.transform import train_loader

model = ResnetModel(num_classes=65)
model.set_optimizer(optim.Adam(model.parameters(), lr=0.001))
model.set_loss_fn(nn.CrossEntropyLoss())
model.set_scheduler(optim.lr_scheduler.StepLR(model.optimizer, step_size=5, gamma=0.1))

for epoch in range(10):
    for x, y in train_loader:
        loss = model.update(x, y)
    model.update_lr()
    print(f"Epoch {epoch+1}, Loss: {loss}")