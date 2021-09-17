from uu.layers import linear 
from uu.layers import base_layer 
import torch


if __name__ == '__main__':
    #create a base class
    l1 = base_layer.BaseLayer().cuda()
    print(l1)
    
    #create a subclass (on GPU)
    print("\n create a operator on GPU device")
    l2 = linear.Linear(10,10).cuda()
    print(l2)
    input = torch.rand(10,10).cuda()
    res = l2(input)
    res = l2(res)

    


    print(res)

    # #create a subclass (on CPU)
    # print("\n create a operator on CPU default")
    # l3 = linear.Linear(10,1)
    # print(l3)
    # input = torch.rand(6,10)
    # res = l3(input)
    # print(res)



    
    
   