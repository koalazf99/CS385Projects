import torchvision
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import os
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage


def tensor_to_PILimage(x):
    unloader = torchvision.transforms.ToPILImage()
    x = unloader(x)
    x = np.asarray(x)
    return x


class CustomDataset(data.Dataset):

    def __init__(self, label, transform=None):
        path = "./data/preprocess_mnist"
        filename = "label_" + str(label) + ".pth"
        self.path = os.path.join(path, filename)
        self.X = torch.load(self.path)
        self.Y = torch.ones(self.X.size()[0]) * label
        tensorint = torch.IntTensor(3)
        self.Y = self.Y.type_as(tensorint)
        self.transform = transform

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, index):
        x = ToPILImage()(self.X[index])
        if self.transform is not None:
            x = self.transform(x)
        sample = [x, self.Y[index]]
        return sample


class PCA(object):
    def __init__(self, data, n_components=2):
        self.x = data
        self.n = n_components
        self.dim = self.x.shape[1]

    def Cov(self):
        x_cov = np.cov(self.x.transpose())
        return x_cov

    def get_feature(self):
        x_cov = self.Cov()
        w, v = np.linalg.eig(x_cov)
        w = w.reshape((v.shape[0], 1))
        unsorted_list = []
        for i, (a, b) in enumerate(zip(w, v)):
            unsorted_list.append({'w': a, 'v': b})
        sorted_list = sorted(unsorted_list, key=lambda k: k['w'], reverse=True)
        return sorted_list

    def reduce_feature(self):
        sorted_list = self.get_feature()
        p = [node['v'][: self.n] for node in sorted_list]
        p = np.asarray(p)
        # print(p.shape)
        y = np.dot(p.transpose(), self.x.transpose())
        y = y.transpose()
        return y


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


if __name__ == "__main__":
    # test Custom Dataset
    dataset = CustomDataset(2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=1)
    for i, data in enumerate(dataloader):
        img = data[0]
        vutils.save_image(img,
                          'test.png',
                          normalize=True)
        print(data[1])
    # --------------------------------------------

    # test PCA
    x = torch.randn([10, 10]).numpy()
    pca = PCA(data=x)
    sorted_list = pca.get_feature()
    y = pca.reduce_feature()
    print(y.shape)
    print(sorted_list)





