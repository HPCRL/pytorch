from uu.layers import linear 
from uu.layers import base_layer 
import torch

if __name__ == '__main__':
   
    l1 = base_layer.BaseLayer().cuda()
    l2 = linear.Linear(1, 10).cuda()
    input = torch.rand(5,10).cuda()
    res = l1(input)
    print(res)

    print(l1)
    print(l2)