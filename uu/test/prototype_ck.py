import torch
from uu.utils import memory 
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter



class Net(torch.nn.Module):
    def __init__(self, m, n, k, l) :
        super(Net, self).__init__()

        # suppose these 2 are streamable
        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l, bias=False) # weight: 840KB  300 700
        # last one is blocked
        self.linear_3 = torch.nn.Linear(l, 1000, bias=False) # weight:  2.8MB 700 1000

        self.stream = torch.nn.Sequential(*[self.linear_1, self.linear_2])

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)
        temp = x[0:99, :]  
        tt = temp.to(device)  # 80K 100 200
        out1 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        temp = x[100:199, :]  
        tt = temp.to(device)  # 80K 100 200
        out2 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        temp = x[200:299, :]  
        tt = temp.to(device)  # 80K 100 200
        out3 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        temp = x[300:399, :]  
        tt = temp.to(device)  # 80K 100 200
        out4 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        blocked_in = torch.cat([out1, out2, out3, out4], dim=0) #1.12M 400 700

        out = self.linear_3(blocked_in) #1.2M 400 1000

        return out


class Net_ref(torch.nn.Module):
    def __init__(self, m, n, k, l, l1_w, l2_w, l3_w) :
        super(Net_ref, self).__init__()

        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l, bias=False) # weight: 840KB  300 700
        self.linear_3 = torch.nn.Linear(l, 1000, bias=False) # weight:  2.8MB 700 1000

        self.linear_1.weight = l1_w
        self.linear_2.weight = l2_w
        self.linear_3.weight = l3_w
    
    def forward(self, x):
        device = self.dummy_param.device
        print("device ", device)
        out = self.linear_3(self.linear_2(self.linear_1(x)))
        return out


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = 400
    n = 300
    k = 200
    l = 700

    model = Net(m, n, k, l).to(device)
    input = torch.rand(m, k, requires_grad = True).cuda() # [400, 200] 320K

    l1_w = model.linear_1.weight
    l2_w = model.linear_2.weight
    l3_w = model.linear_3.weight

    # use same random as Net()
    model_ref = Net_ref(m, n, k, l, l1_w, l2_w, l3_w).to(device)
    model = Net(m, n, k, l).to(device)
    input = torch.rand(m, k, requires_grad = True).cuda() # [400, 200] 320K

    model(input)
    model_ref(input)


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