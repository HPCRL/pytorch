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
    mynet = Network()    
    #mynet.check_compatibility()

    l = mynet.get_first_layer()
    print("first layer : ", l)
    print(l.get_next())
    mynet.traverse(mynet.get_first_layer())

