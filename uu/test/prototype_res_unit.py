import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Residual_ref(nn.Module):  
    """The Residual block of ResNet."""
    def __init__(self, input_channels, output_channels,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()


        #self.sck = torch.nn.Sequential(*[self.conv1, self.bn1, self.relu, self.conv2, self.bn2])

    def forward(self, X):
        s = torch.nn.Sequential(*[self.conv1, self.bn1, self.relu, self.conv2, self.bn2])
        Y = checkpoint(s, X)
        # Y = checkpoint(self.sck, X)
        # Y = self.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        Y += X
        return self.relu(Y)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk1 = Residual_ref(3, 3)
        self.blk2 = Residual_ref(3, 3)
        self.blk3 = Residual_ref(3, 3)


    def forward(self, X, checkpointing=False):
        if not checkpointing:
            Y = self.blk1(X)
            Y = self.blk2(Y)
            Y = self.blk3(Y)
        else:
            s = torch.nn.Sequential(*[self.blk1, self.blk1, self.blk3])
            Y = checkpoint(s, X)
        
        return Y

    




def main():
    blk = Residual_ref(3, 3)
    X = torch.rand(1, 3, 32, 32, requires_grad = True)
    Y = blk(X)
    print(Y.shape)

    # X = torch.rand(1, 3, 32, 32)
    # model = Net()
    # Y = model(X, checkpointing=True)
    # print(Y.shape)



if __name__=="__main__":
    main()


