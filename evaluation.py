import torch
from dataset.transform import transform
from dataset.OfficeHome import OfficeHomeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(epoch, test_loader, test_domain):
    checkpoint = f"checkpoint_{epoch}_test_on_{test_domain}.pth.tar"
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    num_classes = 65
    corrects = [0]*num_classes
    instances = [0]*num_classes
    predictions = [0]*num_classes
    
    accuracy = 0
    precision = [0]*num_classes
    recall = [0]*num_classes
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.tolist()
            
            pred_labels = model(images)
            (_, pred_labels) = torch.max(pred_labels, dim=1)
            pred_labels = pred_labels.tolist()
            
            for i in range(len(labels)):
                instances[labels[i]] += 1
                predictions[pred_labels[i]] += 1
                
                if pred_labels[i] == labels[i]:
                    corrects[labels[i]] += 1
                
    accuracy = sum(corrects) / sum(instances)
    
    for i in range(len(instances)):
        if predictions[i] == 0:
            precision[i] = 0
        else:
            precision[i] = corrects[i] / predictions[i]
        
        recall[i] = corrects[i] / instances[i]
            
    return accuracy, precision, recall

if __name__ == "__main__":
    data_dir = "./dataset/OfficeHomeDataset"
    test_domain = ["Real World"]
    
    test_dataset = OfficeHomeDataset(root_dir=data_dir, domains=test_domain, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    evaluate(0, test_loader, test_domain[0])