import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, label):
        first = [-output[i][label[i]] for i in range(label.size()[0])]
        first_ = 0
        for i in range(len(first)):
            first_ += first[i]

        second = torch.exp(output)
        second = torch.sum(second, dim=1)
        second = torch.log(second + 1e-5)
        second = torch.sum(second)
        loss = 1 / label.size()[0] * (first_ + second)
        return loss


class RidgeLoss(nn.Module):
    def __init__(self):
        super(RidgeLoss, self).__init__()

    def forward(self, beta, bias):
        bias_ = bias.view(1, -1)
        tot = torch.cat((beta, bias_), dim = 0)
        return torch.norm(tot) ** 2


class LassoLoss(nn.Module):
    def __init__(self):
        super(LassoLoss, self).__init__()

    def forward(self, beta, bias):
        bias_ = bias.view(1, -1)
        tot = torch.cat((beta, bias_), dim=0)
        return torch.sum(torch.abs(tot))


if __name__ == '__main__':
    output = torch.randn(3, 5, requires_grad=True)
    label = torch.empty(3, dtype=torch.long).random_(5)
    criterion = nn.CrossEntropyLoss()
    c_2 = CrossEntropyLoss()
    print(criterion(output, label))
    print(c_2(output, label))

    beta = torch.randn(784, 10)
    x = torch.randn(128, )
