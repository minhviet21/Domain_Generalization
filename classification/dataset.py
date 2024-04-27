import torch
import torchvision
from PIL import Image

class OfficeHomeDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root_dir, domain, transform=None):
        super(OfficeHomeDataset, self).__init__(root=root_dir+"/"+domain, transform=transform)
        self.domain = domain

    def get_domain(self):
        return self.domain
    
    def __getitem__(self, idx):
        image_path = self.samples[idx][0]
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.samples[idx][1])
        return image, label