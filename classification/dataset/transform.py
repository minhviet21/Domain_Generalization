from torchvision import transforms
from dataset import OfficeHomeDataset
import torch

data_dir = "/content/drive/MyDrive/OfficeHome"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

art_dataset = OfficeHomeDataset(data_dir, "Art", transform)
clipart_dataset = OfficeHomeDataset(data_dir, "Clipart", transform)
# product_dataset = OfficeHomeDataset(data_dir, "Product", transform)
# real_dataset = OfficeHomeDataset(data_dir, "Real World", transform)

# Iterate through the dataset loader
# train_loader = torch.utils.data.DataLoader(art_dataset, batch_size=32, shuffle=True)

# Merge datasets
merged_dataset = torch.utils.data.ConcatDataset([art_dataset, clipart_dataset])

# Create DataLoader with the merged dataset
train_loader = torch.utils.data.DataLoader(merged_dataset, batch_size=32, shuffle=True)