import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as dataloader
import torch.optim as optim

from ClassificationTask.utils import *
import os


class CNN(nn.Module):
    def __init__(self, num_layers):
        super(CNN, self).__init__()
        assert num_layers == 5, "only support num layer = 5, but got num layer = {}".format(num_layers)
        self.layers = nn.ModuleList()

        # construct conv layers
        in_c = 1
        out_c = 16
        cur_size = 28
        for i in range(0, num_layers - 1):
            if i % 2 == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True)
                    )
                )
                in_c, out_c = out_c, out_c * 2
                cur_size = cur_size - 2
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                )
                in_c, out_c = out_c, out_c * 2
                cur_size = (cur_size - 1) // 2

        self.layers.append(
            nn.Sequential(
                nn.Linear(in_c * cur_size * cur_size, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 10)
            )
        )

        self.in_c = in_c
        self.cur_size = cur_size

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # print(x.size())
        x = x.view(-1, self.in_c * self.cur_size * self.cur_size)
        fc_layer = self.layers[-1]
        x = fc_layer(x)
        return x


num_epochs = 1
# MNIST pic size is 28 * 28
input_dim = 28 * 28
output_dim = 10
lr_rate = 0.05
# GPU setting
GPU = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
batch_size = 128
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))]),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))]),
                              download=True)

train_loader = dataloader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = dataloader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)
x = torch.randn(1, 1, 28, 28)
model = CNN(5)
print(model(x))
model.cuda()

# num_params = 0
# for param in model.parameters():
#     print(param.size())
#     num_params += param.numel()
# print(num_params)

# start training
total_step = len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 1e-1)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()

        # forward propagation
        output = model(images)
        loss = criterion(output, labels.cuda())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# testing
correct = 0
total = 0
for i, (images, labels) in enumerate(test_loader):
    output = model(images.cuda())
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the model on the 10000 test images: {:.2f} %'.format(100.0 * correct / total))








