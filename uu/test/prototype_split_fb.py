import torch
from uu.utils import memory 

from torch.nn.parameter import Parameter

def printsize(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_input size : ', grad_input[0].size())
    print('grad_output size : ', grad_output[0].size())


class Net(torch.nn.Module):
    def __init__(self, m, n, k, l) :

        #[m, k] * [k, n] --> [m, n]*[n, l] --> [m, l]
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(k, n) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l) # weight: 840KB  300 700

        # self.linear_1.weight = Parameter(torch.reshape(torch.arange(0, k*n, step=1.0, dtype=torch.float), (n, k)))
        # self.linear_2.weight = Parameter(torch.reshape(torch.arange(0, -l*n, step=-1.0, dtype=torch.float), (l, n)))  
        # self.linear_1.bias = Parameter(torch.arange(0, n, step=1.0, dtype=torch.float))
        # self.linear_2.bias = Parameter(torch.arange(0, -l, step=-1.0, dtype=torch.float))  


        self.linear_1.register_full_backward_hook(printsize)
        self.linear_2.register_full_backward_hook(printsize)
        
        print("==== linear 1 weight shape", self.linear_1.weight.size())
        print("++++ linear 1 weight ", self.linear_1.weight)
        print("==== linear 2 weight shape", self.linear_2.weight.size())
        print("++++ linear 2 weight ", self.linear_2.weight)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
       

    def forward(self, x):
        device = self.dummy_param.device
        print("device ", device)
        memUsage = memory.MeasureMemory(device)


        #temp = torch.split(x, 100, dim=0) # 320K

        temp0 = torch.narrow(x, dim=0, start=0, length=100)
        temp1 = torch.narrow(x, 0, 100, 300)
        # temp0 = torch.narrow(x, dim=0, start=0, length=1)
        # temp1 = torch.narrow(x, 0, 1, 3)

        # print("==== after split input, will double space?? ...") # looks not
        # print(memUsage.currentValue())
        # print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())
        # print(memUsage.maxx())
        # print(memUsage.currentCached())
        
        result_1 = self.linear_1(temp0)
        # print("==== result_1 shape", result_1.size())
        # print("==== after first Linear ...") # res1 [100*300]
        # print(memUsage.currentValue())
        # print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())
        # print(memUsage.maxx())
        # print(memUsage.currentCached())

        
        result_2 = self.linear_1(temp1)
        # print("==== result_2 shape", result_2.size())
        # print("==== after first Linear ...") # res2 [300*300]        
        # print(memUsage.currentValue())
        # print(memUsage.availableValue())
        # print(memUsage.current())
        # print(memUsage.available())
        # print(memUsage.maxx())
        # print(memUsage.currentCached())

        f_res1 = self.linear_2(result_1) #[100*700]
        f_res2 = self.linear_2(result_2) #[300*700]
        out_list = [f_res1, f_res2]
        out = torch.cat(out_list, dim=0)
        return out



class Net_ref(torch.nn.Module):
    def __init__(self, m, n, k, l, l1_w, l2_w, l1_b, l2_b) :

        #[m, k] * [k, n] --> [m, n]*[n, l] --> [m, l]
        super(Net_ref, self).__init__()
        self.linear_1 = torch.nn.Linear(k, n) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l) # weight: 840KB  300 700

        # self.linear_1.weight = Parameter(torch.reshape(torch.arange(0, k*n, step=1.0, dtype=torch.float), (n, k)))
        # self.linear_2.weight = Parameter(torch.reshape(torch.arange(0, -l*n, step=-1.0, dtype=torch.float), (l, n)))  
        # self.linear_1.bias = Parameter(torch.arange(0, n, step=1.0, dtype=torch.float))
        # self.linear_2.bias = Parameter(torch.arange(0, -l, step=-1.0, dtype=torch.float))  

        self.linear_1.weight = l1_w
        self.linear_2.weight = l2_w
        self.linear_1.bias = l1_b
        self.linear_2.bias = l2_b

        self.linear_1.register_full_backward_hook(printsize)
        self.linear_2.register_full_backward_hook(printsize)
        
        print("==== ref linear 1 weight shape", self.linear_1.weight.size())
        print("++++ ref linear 1 weight ", self.linear_1.weight)
        print("==== ref linear 2 weight shape", self.linear_2.weight.size())
        print("++++ ref linear 2 weight ", self.linear_2.weight)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
       

    def forward(self, x):
        device = self.dummy_param.device
        print("device ", device)
        memUsage = memory.MeasureMemory(device)

        out = self.linear_2(self.linear_1(x))

        return out

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    # print("==== init ...")
    # print(memUsage.currentValue())
    # print(memUsage.availableValue())
    # print(memUsage.current())
    # print(memUsage.available())
    # print(memUsage.maxx())
    # print(memUsage.currentCached())

    m = 400
    n = 300
    k = 200
    l = 700

    # m = 4
    # n = 3
    # k = 2
    # l = 7
    model = Net(m, n, k, l).to(device)
    #model_ref = Net_ref(m, n, k, l).to(device)

    l1_w = model.linear_1.weight
    l2_w = model.linear_2.weight
    l1_b = model.linear_1.bias
    l2_b = model.linear_2.bias
    # use same random as Net()
    model_ref = Net_ref(m, n, k, l, l1_w, l2_w, l1_b, l2_b).to(device)



    # print("==== after define network (weight size) ...")
    # print(memUsage.currentValue())
    # print(memUsage.availableValue())
    # print(memUsage.current())
    # print(memUsage.available())
    # print(memUsage.maxx())
    # print(memUsage.currentCached())



    input = torch.rand(m, k, requires_grad = True).cuda() # 320K
    # print("==== after allocate input ...")
    # print(memUsage.currentValue())
    # print(memUsage.availableValue())
    # print(memUsage.current())
    # print(memUsage.available())
    # print(memUsage.maxx())
    # print(memUsage.currentCached())


    out = model(input)
    out_ref = model(input)
    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("check forward correctness")
    assert(torch.all(torch.eq(out, out_ref)))

    out.sum().backward()
    out_ref.sum().backward()
    
    out_w_list = []
    i = 0
    for name, param in model.named_parameters():
        print("i", i, name, param.size())
        i += 1
        out_w_list.append(param)

    out_ref_w_list = []
    i = 0
    for name, param in model_ref.named_parameters():
        print("i", i, name, param.size())
        i += 1
        out_ref_w_list.append(param)


    for ww in range(0, i):
        print("out_w {} \n out_ref_w {}".format(out_w_list[ww], out_ref_w_list[ww]))
        assert(torch.all(torch.eq(out_w_list[ww], out_ref_w_list[ww])))

if __name__=="__main__":
    main()
