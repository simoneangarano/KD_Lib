import torch

class KLDivLoss(torch.nn.Module):
    def __init__(self, T=1.0):
        super(KLDivLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.T = T

    def forward(self, outputs, targets):
        outputs = self.log_softmax(outputs)
        targets = self.softmax(targets)
        return torch.nn.KLDivLoss()(outputs, targets)