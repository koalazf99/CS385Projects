import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as dataloader
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
import os


class LogisticRegression_no_frame(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_no_frame, self).__init__()
        # self.beta = nn.Parameter(input_dim, output_dim)
        # self.bias = nn.Parameter(output_dim)
        self.beta_ = torch.randn(input_dim, output_dim)
        self.bias_ = torch.randn(output_dim)
        self.beta = nn.Parameter(self.beta_)
        self.bias = nn.Parameter(self.bias_)
    def forward(self, x):
        output = torch.mm(x, self.beta)
        for i in range(output.size()[0]):
            output[i] += self.bias
        return output
    def update(self, lr):
        # print(self.beta.grad, self.bias.grad)
        self.beta_ = self.beta.data - lr * self.beta.grad.data
        self.bias_ = self.bias.data - lr * self.bias.grad.data
        self.beta.data = self.beta_
        self.bias.data = self.bias_

# hyperParameter configuration
batch_size = 128
num_epochs = 5
# MNIST pic size is 28 * 28
input_dim = 28 * 28
output_dim = 10
lr_rate = 0.03
# GPU setting
GPU = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU


train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]),
                              download=True)

train_loader = dataloader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = dataloader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# define model
model = LogisticRegression_no_frame(input_dim, output_dim)
model = model.cuda()

# start training
total_step = len(train_loader)
criterion = CrossEntropyLoss()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        input_size = images.size()[1] * images.size()[2] * images.size()[3]
        images = images.view(-1, input_size).cuda()
        images = images / 255.0
        output = model(images)
        # compute loss, backward
        loss = criterion(output, labels.cuda())
        # print(loss)
        loss.backward()
        # update parameter use gradient descent method
        model.update(lr_rate)

        # print(model_beta)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# testing
correct = 0
total = 0
for i, (images, labels) in enumerate(test_loader):
    input_size = images.size()[1] * images.size()[2] * images.size()[3]
    images = images.view(-1, input_size)
    # compute = y = x^T * \beta + bias, and then use softmax to do prediction
    output = model(images.cuda())
    output = F.softmax(output, dim=1)
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
