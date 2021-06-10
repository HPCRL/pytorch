import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.datasets.cifar import CIFAR10
import numpy as np
from progressbar import progressbar


def conv_bn_relu(in_ch, out_ch, ker_sz, stride, pad):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU())


class NetA(nn.Module):
    def __init__(self, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        k = 2
        # 32x32
        self.layer1 = conv_bn_relu(3, 32*k, 3, 1, 1)
        self.layer2 = conv_bn_relu(32*k, 32*k, 3, 2, 1)
        # 16x16
        self.layer3 = conv_bn_relu(32*k, 64*k, 3, 1, 1)
        self.layer4 = conv_bn_relu(64*k, 64*k, 3, 2, 1)
        # 8x8
        self.layer5 = conv_bn_relu(64*k, 128*k, 3, 1, 1)
        self.layer6 = conv_bn_relu(128*k, 128*k, 3, 2, 1)
        # 4x4
        self.layer7 = conv_bn_relu(128*k, 256*k, 3, 1, 1)
        self.layer8 = conv_bn_relu(256*k, 256*k, 3, 2, 1)
        # 1x1
        self.layer9 = nn.Linear(256*k, 10)

    def seg0(self, y):
        y = self.layer1(y)
        return y

    def seg1(self, y):
        y = self.layer2(y)
        y = self.layer3(y)
        return y

    def seg2(self, y):
        y = self.layer4(y)
        y = self.layer5(y)
        return y

    def seg3(self, y):
        y = self.layer6(y)
        y = self.layer7(y)
        return y

    def seg4(self, y):
        y = self.layer8(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.layer9(y)
        return y

    def forward(self, x):
        y = x
        # y = self.layer1(y)
        y = y + torch.zeros(1, dtype=y.dtype, device=y.device, requires_grad=True)
        if self.use_checkpoint:
            # Use checkpoint
            y = checkpoint(self.seg0, y)
            y = checkpoint(self.seg1, y)
            y = checkpoint(self.seg2, y)
            y = checkpoint(self.seg3, y)
            y = checkpoint(self.seg4, y)
        else:
            # Do not use checkpoint
            y = self.seg0(y)
            y = self.seg1(y)
            y = self.seg2(y)
            y = self.seg3(y)
            y = self.seg4(y)

        return y


if __name__ == '__main__':
    net = NetA(use_checkpoint=True).cuda()

    train_dataset = CIFAR10('../datasets/cifar10', True, download=True)
    train_x = np.asarray(train_dataset.data, np.uint8)
    train_y = np.asarray(train_dataset.targets, np.int)

    losser = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), 1e-3)

    epoch = 10
    batch_size = 31
    batch_count = int(np.ceil(len(train_x) / batch_size))

    for e_id in range(epoch):
        print('epoch', e_id)

        print('training')
        net.train()
        loss_sum = 0
        for b_id in progressbar(range(batch_count)):
            optim.zero_grad()

            batch_x = train_x[batch_size*b_id: batch_size*(b_id+1)]
            batch_y = train_y[batch_size*b_id: batch_size*(b_id+1)]

            batch_x =  torch.from_numpy(batch_x).permute(0, 3, 1, 2).float() / 255.
            batch_y =  torch.from_numpy(batch_y).long()

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            batch_x = F.interpolate(batch_x, (224, 224), mode='bilinear')

            y = net(batch_x)
            loss = losser(y, batch_y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
        print('loss', loss_sum / batch_count)

        with torch.no_grad():
            print('testing')
            net.eval()
            acc_sum = 0
            for b_id in progressbar(range(batch_count)):
                optim.zero_grad()

                batch_x = train_x[batch_size * b_id: batch_size * (b_id + 1)]
                batch_y = train_y[batch_size * b_id: batch_size * (b_id + 1)]

                batch_x = torch.from_numpy(batch_x).permute(0, 3, 1, 2).float() / 255.
                batch_y = torch.from_numpy(batch_y).long()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                batch_x = F.interpolate(batch_x, (224, 224), mode='bilinear')

                y = net(batch_x)

                y = torch.topk(y, 1, dim=1).indices
                y = y[:, 0]

                acc = (y == batch_y).float().sum() / len(batch_x)

                acc_sum += acc.item()
            print('acc', acc_sum / batch_count)

        ids = np.arange(len(train_x))
        np.random.shuffle(ids)
        train_x = train_x[ids]
        train_y = train_y[ids]


