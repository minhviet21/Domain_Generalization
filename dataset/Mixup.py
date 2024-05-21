import torch
import numpy as np
import torch.nn.functional as F

class DataAugmentation():
  def __init__(self,a,b,minibatches,num_classes):
    self.a=a
    self.b=b
    self.minibatches=minibatches
    self.num_classes=num_classes
  # def indices_to_onehot(self,indices):
  #   onehot=F.one_hot(indices, self.num_classes)
  #   return onehot
  def random_pairs_of_minibatches(self):
    perm = torch.randperm(len((self.minibatches)[0])).tolist()
    pairs = []

    for i in range(len((self.minibatches)[0])):
        j = i + 1 if i < (len((self.minibatches)[0]) - 1) else 0

        xi, yi = self.minibatches[0][perm[i]], self.minibatches[1][perm[i]]
        xj, yj = self.minibatches[0][perm[j]], self.minibatches[1][perm[j]]

        if xi.ndimension() == 0:
            xi = xi.unsqueeze(0)
        if xj.ndimension() == 0:
            xj = xj.unsqueeze(0)
        if yi.ndimension() == 0:
            yi = yi.unsqueeze(0)
        if yj.ndimension() == 0:
            yj = yj.unsqueeze(0)

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
    return pairs
  def Augmentation(self):
    batch_x=[]
    batch_y=[]
    for (xi, yi), (xj, yj) in self.random_pairs_of_minibatches():
      lam = np.random.beta(self.a,self.b)

      x = lam * xi + (1 - lam) * xj
      # y=  lam * yi + (1-lam) *yj
      batch_x.append(x)
      batch_y.append(torch.Tensor([yi,yj]))
    batch_x=torch.stack(batch_x)
    batch_y=torch.stack(batch_y)

    return list((batch_x,batch_y))

def NewData(a=0.2, b=0.2, num_classes=65, loader=None):
    if loader is None:
        raise ValueError("loader must be provided")
    train_batches = []
    for minibatches in loader:
        data_augmentor = DataAugmentation(a, b, minibatches, num_classes)
        batch = data_augmentor.Augmentation()
        train_batches.append(batch)
    return train_batches