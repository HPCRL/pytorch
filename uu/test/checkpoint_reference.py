import torch
from uu.utils import memory 
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter

class Net(torch.nn.Module):
    def __init__(self, m, n, k, l) :
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l, bias=False) # weight: 840KB  300 700
        self.linear_3 = torch.nn.Linear(l, l, bias=False) # weight:  ~1.9 MB 700 700
        self.linear_4 = torch.nn.Linear(l, 1000, bias=False) # weight:  2.8MB 700 1000
        self.linear_5 = torch.nn.Linear(1000, 2000, bias=False) # weight:  8MB 1000 2000

        self.li3 = torch.nn.Sequential(*[self.linear_1, self.linear_2, self.linear_3, self.linear_4, self.linear_5])

        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        

    

    def forward(self, x, checkpointing=True):    # [400, 200] 320K
        device = self.dummy_param.device
        print("device ", device)
        memUsage = memory.MeasureMemory(device)
        x = checkpoint(self.li3, x)
        # if checkpointing:
        #     print("==== init ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #init now should be around 6MB+8
        #     print(memUsage.current())
        #     x = self.linear_1(x)                #480K [400, 300] 
        #     print("==== 1st lin ...")
        #     print(memUsage.snapshot())          #+480K = 6.5M+8
        #     print(memUsage.currentValue())      
        #     print(memUsage.current())


        #     x = checkpoint(self.li3, x)         # should not save out2, out3 (1.12 x 2)
        #                                         #out 1.6M [400, 1000]
        #     print("==== checkp 3 lin ...")
        #     print(memUsage.snapshot())          
        #     print(memUsage.currentValue())      #+1.6M = 8.1M+8
        #     print(memUsage.current())

        #     x = self.linear_5(x)                #3.2M [400, 2000]
        #     print("==== last lin ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+3.2M = 11.3+8
        #     print(memUsage.current())
        # else:
        #     print("==== init ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #init now should be around 6MB+8
        #     print(memUsage.current())
            
        #     x = self.linear_1(x)
        #     print("==== 1st lin ...")           #480K [400, 300]
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+480K = 6.5M+8
        #     print(memUsage.current())
            
        #     x = self.linear_2(x)                #1.12M [400, 700]
        #     print("==== 2nd lin ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+1.12M = 7.6M+8
        #     print(memUsage.current())
            
        #     x = self.linear_3(x)                #1.12M [400, 700]
        #     print("==== 3rd lin ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+1.12M = 8.7M+8
        #     print(memUsage.current())
            
        #     x = self.linear_4(x)                #1.6M [400, 1000]
        #     print("==== 4th lin ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+1.6M = 10.3 M+8
        #     print(memUsage.current())

        #     x = self.linear_5(x)                #3.2M [400, 2000]
        #     print("==== 5th lin ...")
        #     print(memUsage.snapshot())
        #     print(memUsage.currentValue())      #+3.2M = 13.5 M+8
        #     print(memUsage.current())

        return x

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    m = 400
    n = 300
    k = 200
    l = 700

    model = Net(m, n, k, l).to(device)
    input = torch.rand(m, k, requires_grad = True).cuda() # [400, 200] 320K
    out = model(input, False)
    print("--- print(out)", out)
    out.sum().backward()



if __name__=="__main__":
    main()


