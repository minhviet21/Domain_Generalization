from torch import nn
import torch.optim as optim
from model import ResnetModel
from dataset import train_loader

model = ResnetModel(num_classes=65)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def update():
  for images, labels in train_loader:
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(10):
    loss_ = update()
    print(f"Epoch {epoch + 1}: Training loss = {loss_:.4f}")