from uu.layers import linear 
from uu.layers import base_layer 
import torch

if __name__ == '__main__':
    #create a linear chain
    l0 = linear.Linear(10,1).cuda()
    print(l0)
    
    l1 = linear.Linear(13,21)
    print(l1)

    l2 = linear.Linear(3,66)
    print(l2)
    
    l0.hook(l1) #   l0(l1(Input))
    l1.hook(l2)

 

    l0_next = l0.get_next()
    print("l0_next ", l0_next)
    l0_prev = l0.get_prev()
    print("l0_prev ", l0_prev)

    l1_next = l1.get_next()
    print("l1_next ", l1_next)
    l1_prev = l1.get_prev()
    print("l1_prev ", l1_prev)

    l2_next = l2.get_next()
    print("l2_next ", l2_next)
    l2_prev = l2.get_prev()
    print("l2_prev ", l2_prev)




    