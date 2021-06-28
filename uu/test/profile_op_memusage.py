from uu.layers import linear 
from uu.layers import base_layer 
from uu.layers import base_network 
import torch
from torch import Tensor
from uu.utils import memory 


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    
    print("==== init ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())

    linear_t = torch.nn.Linear(10, 1000).to(device)

    print("==== after define op (weight size) ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())


    input = torch.rand(3000, 10).cuda()
    #input.requires_grad = True

    print("==== after allocate input ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())


    output = linear_t(input)

    print("==== after compute output ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())


    erro = output.sum()
    erro.backward()
    print("==== call backward ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())

    #print (input.grad)
