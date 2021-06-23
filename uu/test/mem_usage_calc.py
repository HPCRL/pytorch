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
        linear1 = linear.Linear(10,1)
        linear2 = linear.Linear(1, 77)
        linear3 = linear.Linear(77,100)
        linear4 = linear.Linear(100, 5)

        #self.first_layer = linear1
        self.set_first_layer(linear1)

        #constrcut network
        linear1.hook(linear2)
        linear2.hook(linear3)
        linear3.hook(linear4)
    

class Network1(base_network.BaseNetwork):
    def __init__(self):
        super(Network1, self).__init__()
        #define ops
        linear1 = linear.Linear(10,1)
        linear2 = linear.Linear(1, 77)
        linear3 = linear.Linear(77,100)
        

        #self.first_layer = linear1
        self.set_first_layer(linear1)

        #constrcut network
        linear1.hook(linear2)
        linear1.hook(linear3)


class Network2(base_network.BaseNetwork):
    def __init__(self):
        super(Network2, self).__init__()
        #define ops
        linear1 = linear.Linear(10,1)
        linear2 = linear.Linear(1, 77)
        linear3 = linear.Linear(77,100)
        linear4 = linear.Linear(100, 5)

        #self.first_layer = linear1
        self.set_first_layer(linear1)

        #constrcut network
        linear1.hook(linear2)
        linear1.hook(linear3)
        linear2.hook(linear4)
        linear3.hook(linear4)


class Network3(base_network.BaseNetwork):
    def __init__(self):
        super(Network3, self).__init__()
        #define ops
        linear1 = linear.Linear(10,1)
        linear2 = linear.Linear(1, 77)
        linear3 = linear.Linear(77,100)
        linear4 = linear.Linear(100, 5)
        linear5 = linear.Linear(5, 12)

        #self.first_layer = linear1
        self.set_first_layer(linear1)

        #constrcut network
        linear1.hook(linear2)
        linear1.hook(linear3)
        linear2.hook(linear4)
        linear3.hook(linear4)
        linear4.hook(linear5)
       

class Network4(base_network.BaseNetwork):
    def __init__(self):
        super(Network4, self).__init__()
        #define ops
        linear1 = linear.Linear(100,1)
        linear2 = linear.Linear(1, 77)
        linear3 = linear.Linear(77,100)
        linear4 = linear.Linear(17,100)
        linear5 = linear.Linear(33,1020)


        #self.first_layer = linear1
        self.set_first_layer(linear1)

        #constrcut network
        """
        l1--l4--l3--\ 
          \         ---l5  
           \-l2-----/       
        """
        linear1.hook(linear4)
        linear1.hook(linear2)
        linear4.hook(linear3)
        linear4.hook(linear5)
        linear2.hook(linear5)

if __name__ == '__main__':
    # test single layer mem_usage
    mynet = Network()    
    l = mynet.get_first_layer()
    print("first layer : ", l)
    print(l.get_next())

    input = ft.FakeTensor([5, 1000])
    # # single op mem_usage
    # l.mem_usage(input)

    # # test a linear network mem_usage
    # mynet.mem_usage(input)

    # # test a 2 way fork network mem_usage
    # mynet_fork = Network1()
    # mynet_fork.mem_usage(input)

    # # test a diamond network mem_usage
    # mynet_diamond = Network2()
    # mynet_diamond.mem_usage(input)

    # # test a diamond network with tail mem_usage
    # mynet_diamond_t = Network3()
    # mynet_diamond_t.mem_usage(input)

    # test a random fan-out network with tail mem_usage
    mynet_r = Network4()
    mynet_r.mem_usage(input)


