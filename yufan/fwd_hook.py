import torch
import torch.nn as nn

class LinearTransformation(nn.Module):
    def __init__(self, constant=False):
        super(LinearTransformation, self).__init__()
        self.linear = nn.Linear(2, 4)
       
    
    def forward(self, x):
        return torch.norm(self.linear(x)) ** 2

def hook(self, input):
    print(input)
model = LinearTransformation(constant=False)


def main():
    x = torch.tensor([1., 2.])

    model.linear.register_forward_pre_hook(hook)
    out = model(x)
    print(out)

if __name__=="__main__":
    main()
