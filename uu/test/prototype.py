import torch
from uu.utils import memory 

def printsize(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    print('grad_input : ', len(grad_input))
    print('grad_output : ', len(grad_output))
    print('grad_input size : ', grad_input[0].size())
    print('grad_output size : ', grad_output[0].size())


class Net(torch.nn.Module):
    def __init__(self, m, n, k, l) :

        #[m, k] * [k, n] --> [m, n]*[n, l] --> [m, l]
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(k, n) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l) # weight: 840KB  300 700

        self.linear_1.register_full_backward_hook(printsize)
        self.linear_2.register_full_backward_hook(printsize)
        
        print("==== linear 1 weight shape", self.linear_1.weight.size())
        print("==== linear 2 weight shape", self.linear_2.weight.size())
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
       


    def forward(self, x):
        device = self.dummy_param.device
        print("device ", device)
        memUsage = memory.MeasureMemory(device)


        #temp = torch.split(x, 100, dim=0) # 320K
        temp0 = torch.narrow(x, dim=0, start=0, length=100)
        temp1 = torch.narrow(x, 0, 100, 300)

        print("==== after split input, will double space?? ...") # looks not
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())
        
        result_1 = self.linear_1(temp0)
        print("==== result_1 shape", result_1.size())
        print("==== after first Linear ...") # res1 [100*300]
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())

        
        result_2 = self.linear_1(temp1)
        print("==== result_2 shape", result_2.size())
        print("==== after first Linear ...") # res2 [300*300]        
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())

        f_res1 = self.linear_2(result_1) #[100*700]
        f_res2 = self.linear_2(result_2) #[300*700]
        out_list = [f_res1, f_res2]
        out = torch.cat(out_list, dim=0)
        return out

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    print("==== init ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())

    m = 400
    n = 300
    k = 200
    l = 700
    model = Net(m, n, k, l).to(device)

    print("==== after define network (weight size) ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())



    input = torch.rand(m, k, requires_grad = True).cuda() # 320K
    print("==== after allocate input ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())



    # for module in model.modules():
    #     #module.register_full_backward_hook(printsize)
    #     print('added hook to', module)

    out = model(input)
    print("out shape", out.size())

    out.sum().backward()
    
    #(out[0].sum() + out[1].sum()).backward()
    # for name, param in model.named_parameters():
    #     print(name, param.grad)


if __name__=="__main__":
    main()
