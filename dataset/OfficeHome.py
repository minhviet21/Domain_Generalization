import os
import torch
import torchvision
from PIL import Image

class OfficeHomeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, domains, transform=None):
        self.samples = []
        self.transform = transform
        self.domains = domains

        for domain in domains:
            domain_path = os.path.join(root_dir, domain)
            if not os.path.isdir(domain_path):
                raise ValueError(f"Domain directory '{domain_path}' not found.")
            
            domain_dataset = torchvision.datasets.ImageFolder(root=domain_path, transform=transform)
            self.samples.extend(domain_dataset.samples)
    def get_domains(self):
        return self.domains
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)    
        return image, label