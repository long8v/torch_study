import torch
from torch import nn

class LabelSmoothingLoss_aftersoftmax(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index = 1):
        super(LabelSmoothingLoss_aftersoftmax, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ignore_indices = pred != self.ignore_index
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.masked_fill_(ignore_indices == 0, self.smoothing / (self.cls - 1 - 1)) #mask 된 건 0으로 남아있음
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))