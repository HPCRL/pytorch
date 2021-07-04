import torch
from uu.utils import memory 

from torch.nn.parameter import Parameter

def printsize(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_input size : ', grad_input[0].size())
    print('grad_output size : ', grad_output[0].size())


class test_Net_ref(torch.nn.Module):
    def __init__(self) :
        super(test_Net_ref, self).__init__()
        self.linear = torch.nn.Linear(1000,500)  # 2MB
        
    
    def forward(self, x):
        #with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)
        print("==== init ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be around 2MB
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())

        temp = x[0:19, :]        
        #print("temp", temp)
        
        print("==== ref slicing ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())       #shoud no change
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())

        tt = temp.to(device)  # 20x1000 80KB
        

        print("==== send slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #shoud +80K
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())

        #print("send temp to cuda", temp)
        out1 = self.linear(tt)    # 20 x 500 40KB
        #print("out", out1)
        print("==== output slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #should +40KB
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())


        
        del tt    
        #del out1
        print("==== after del temp buffer ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #should -80KB
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())


        temp = x[20:39, :]
        print("==== 2nd slicing ...")
        print(memUsage.currentValue())       #shoud no change
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())

        tt = temp.to(device='cuda')  # # 20x1000 80KB
        print("==== send 2nd slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #shoud +80K
        print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())

        out2 = self.linear(tt)
        print("==== output slice ...")
        print(memUsage.snapshot())
        del tt


        print("==== final ...")
        del out1
        del out2
        print(memUsage.snapshot())


        


def main():
    input = torch.rand(100, 1000, requires_grad = True)
    model = test_Net_ref().cuda()

    model(input)

if __name__=="__main__":
    main()
