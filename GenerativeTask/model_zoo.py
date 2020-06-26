import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class VAE_mnist(nn.Module):
    def __init__(self, zsize):
        super(VAE_mnist, self).__init__()

        self.fc11 = nn.Linear(1024, 400)
        self.fc12 = nn.Linear(400, 400)
        self.fc13 = nn.Linear(400, 200)

        # self.relu1 = nn.ReLU()

        self.fc21 = nn.Linear(200, zsize)
        self.fc22 = nn.Linear(200, zsize)

        self.fc3 = nn.Linear(zsize, 200)
        self.fc4 = nn.Linear(200, 400)
        self.fc5 = nn.Linear(400, 1024)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc13(self.fc12(self.fc11(x))))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.fc4(h3)
        x = self.fc5(h3)
        x = x.view(x.size(0), 1, 32, 32)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_relu_feature(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc13(self.fc12(self.fc11(x))))
        return h1

    def get_fc13_feature(self, x):
        x = x.view(x.size(0), -1)
        z = self.fc13(self.fc12(self.fc11(x)))
        return z

    def get_fc12_feature(self, x):
        x = x.view(x.size(0), -1)
        z = self.fc12(self.fc11(x))
        return z

    def get_repara_feature(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z


class VAE_cifar(nn.Module):
    def __init__(self):
        super(VAE_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.fc1 = nn.Linear(1024 * 12, 1200)
        # self.relu1 = nn.ReLU()

        self.fc21 = nn.Linear(1200, 120)
        self.fc22 = nn.Linear(1200, 120)

        self.fc3 = nn.Linear(120, 1200)
        self.fc4 = nn.Linear(1200, 1024 * 12)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1,
                               padding=1, bias=True)

    def encode(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.fc4(h3)
        x = h3.view(h3.size(0), 12, 32, 32)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


import torch
from torch import nn
from torch.nn import functional as F


class VAE_cifar_new(nn.Module):
    def __init__(self, zsize=100, layer_count=3, channels=3):
        super(VAE_cifar_new, self).__init__()
        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class VAE_cifar_gan(nn.Module):
    def __init__(self, nc=3, ndf=64, nz=100):
        super(VAE_cifar_gan, self).__init__()
        self.encode_module = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc1 = nn.Linear(ndf * 8 * 4, nz)
        self.fc2 = nn.Linear(ndf * 8 * 4, nz)
        self.zsize = nz

        self.decode_module = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )


    def encode(self, x):
        x = self.encode_module(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = self.decode_module(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class VAE_vgg(nn.Module):
    def __init__(self):
        super(VAE_vgg, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        # conv size : 1, 64, 4, 4
        self.fc1 = nn.Linear(1 * 64 * 4 * 4, 512)
        self.fc21 = nn.Linear(512, 256)
        self.fc22 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 256)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.downsample(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), -1, 4, 4)
        x = self.upsample(z)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

# For DCGAN config
class Generator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))
        self.dense = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            output = self.dense(output)

        return output.view(-1, 1).squeeze(1)

    def get_feature(self, x):
        output = self.main(x)
        return output


class Discriminator_orig(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator_orig, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            output = self.dense(output)

        return output.view(-1, 1).squeeze(1)

    def get_feature(self, x):
        output = self.main(x)
        return output



if __name__ == '__main__':

    # test mnist VAE model
    test_model = VAE_mnist(zsize=20)
    test_model.cuda()
    x = torch.randn(16, 1, 32, 32).cuda()
    x_vae, mu, logvar = test_model(x)
    print(x_vae.size())

    # test cifar10 VAE model
    test_model = VAE_cifar()
    test_model.cuda()
    x = torch.randn(16, 3, 32, 32).cuda()
    x_vae, mu, logvar = test_model(x)
    print(x_vae.size())

    # test vgg cifar10 VAE model
    test_model = VAE_vgg()
    test_model.cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    x_vae, mu, logvar = test_model(x)
    print(x_vae.size())

    # test dcgan model
    test_model = Discriminator(ngpu=1, nc=3, ndf=64)
    test_model = test_model.cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    out = test_model(x)
    z = test_model.get_feature(x)
    print(out.size(), z.size())