import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as dataloader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input data dimension
        :param output_dim: output data dimension,
        for classification, it means category numbers
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=output_dim,
                                bias=True)

    def forward(self, x):
        output = self.linear(x)
        return output


# hyperParameter configuration
batch_size = 128
num_epochs = 5
# MNIST pic size is 28 * 28
input_dim = 28 * 28
output_dim = 10
lr_rate = 0.001
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

model = LogisticRegression(input_dim, output_dim)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_rate)

# start training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        input_size = images.size()[1] * images.size()[2] * images.size()[3]
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images.cuda())
        loss = criterion(outputs, labels.cuda())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        input_size = images.size()[1] * images.size()[2] * images.size()[3]
        images = images.reshape(-1, input_size)
        outputs = model(images.cuda())
        labels = labels.cuda()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
