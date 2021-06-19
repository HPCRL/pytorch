from uu.layers import linear 
from uu.layers import base_layer 
import torch

if __name__ == '__main__':
    #create a base class
    l1 = base_layer.BaseLayer().cuda()
    print(l1)
    
    #create a subclass (on GPU)
    l2 = linear.Linear(10,1).cuda()
    print(l2)
    input = torch.rand(5,10).cuda()
    res = l2(input)
    print(res)

    #create a subclass (on CPU)
    l3 = linear.Linear(10,1)
    print(l3)
    input = torch.rand(6,10)
    res = l3(input)
    print(res)



    
    
   