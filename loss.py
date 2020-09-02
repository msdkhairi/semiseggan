import torch
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, ignore_index=0, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()

        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        loss = self.criterion(input, target)
        return loss


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        

    def forward(self, input, target, target_mask=None):
        
        if target_mask is not None:
            target = target[target_mask]
            input = input[target_mask]

        loss = self.criterion(input, target)
        return loss




if __name__ == "__main__":
    
    loss = CrossEntropyLoss2d()
    batch_size = 8
    num_classes = 38
    h, w = 350, 350
    input = torch.randn(batch_size, num_classes, h, w, requires_grad=True)
    target = torch.empty(batch_size, h, w, dtype=torch.long).random_(num_classes)
    output = loss(input, target)
    print(output)