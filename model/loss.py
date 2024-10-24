import torch
import torch.nn as nn

class MyCrossEntrpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, pred, target):
        pred = self.log_softmax(pred)
        return -torch.sum(target * pred) / len(target)