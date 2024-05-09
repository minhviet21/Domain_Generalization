import torch
from torch import nn
from model.ResnetBase import ResnetBase
from model.ResnetRSC import ResnetRSC
from dataset.transform import transform
from dataset.OfficeHome import OfficeHomeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(epoch, test_loader, test_domain):
    checkpoint = f"checkpoint_{epoch}_test_on_{test_domain}.pth.tar"
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    
    model.eval()
    
    corrects = []
    instances = []
    predictions = []
    
    accuracy = 0
    precision = []
    recall = []
    
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            label = int(label.item())
            pred_labels = model(image)
            pred_label = torch.max(pred_labels, dim=2)
            pred_label = int(pred_label.item())
            
            instances[label] += 1
            predictions[pred_label] += 1
            
            if pred_label == label:
                corrects[label] += 1
                
    accuracy = sum(corrects) / sum(instances)
    for i in range(len(instances)):
        precision[i] = corrects[i] / predictions[i]
        recall[i] = corrects[i] / instances[i]
            
    return accuracy, precision, recall