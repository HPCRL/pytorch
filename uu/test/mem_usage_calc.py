from uu.layers import linear 
from uu.layers import base_layer 
from uu.layers import base_network 
import torch
from torch import Tensor
from uu.utils import ftensor as ft

class Network(base_network.BaseNetwork):
    def __init__(self):
        super(Network, self).__init__()
        #define ops
        linear1 = linear.Linear(100000000000000000,1)
        linear2 = linear.Linear(1, 770000000)
        linear3 = linear.Linear(770000000,100)

        #constrcut network
        linear1.hook(linear2)
        linear2.hook(linear3)



 



if __name__ == '__main__':
    input = ft.FakeTensor([7700, 100000000000000000])
    mynet = Network()
    mynet.set_first_layer(mynet.linear1)
    mem_cost = mynet.mem_usage(input)

    print(mem_cost)



