import torch
from uu.utils import memory 
from uu.utils import correctness_check 
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter


class Residual_lin_ref(torch.nn.Module):  
    """The Residual block of ResNet."""
    def __init__(self, n, k):
        super().__init__()
        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, k, bias=False) # weight: 240KB  300 200
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)

        print("==== before ref ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())     
        print(memUsage.availableValue())
        Y = self.relu(self.linear_2(self.linear_1(X)))
        Y += X
        print("==== after ref ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      
        print(memUsage.availableValue())

        return Y


class Residual_lin(torch.nn.Module):  
    """The Residual block of ResNet."""
    def __init__(self, n, k, sp_l):
        super().__init__()
        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, k, bias=False) # weight: 240KB  300 200
        self.relu = torch.nn.ReLU()
        self.sp_l = sp_l
        self.stream = torch.nn.Sequential(*[self.linear_1, self.linear_2, self.relu])

    def forward(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)

        print("==== before split ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be around 480KB x 2
        print(memUsage.availableValue())

        temp = X[0:self.sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        #out1 = checkpoint(self.stream, tt)      #80K 100 200
        out1 = self.stream(tt)
        out1 += tt

        del tt

        print("==== after 1st slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be + 120K
        print(memUsage.availableValue())

        temp = X[self.sp_l:2*self.sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        #out2 = checkpoint(self.stream, tt)      
        out2 = self.stream(tt)
        out2 += tt
        del tt

        print("==== after 2nd slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      
        print(memUsage.availableValue())

        temp = X[2*self.sp_l:3*self.sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        #out3 = checkpoint(self.stream, tt)      
        out3 = self.stream(tt)
        out3 += tt
        del tt

        print("==== after 3rd slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      
        print(memUsage.availableValue())


        temp = X[3*self.sp_l:4*self.sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        #out4 = checkpoint(self.stream, tt)     
        out4 = self.stream(tt)
        out4 += tt
        del tt

        print("==== after 4th slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())     
        print(memUsage.availableValue())


        # now should be around 480K + 320K
        final_out = torch.cat([out1, out2, out3, out4], dim=0) #320K 400 200
        del out1
        del out2
        del out3
        del out4
        print("==== after concatenate slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should not increased
        print(memUsage.availableValue())

        return final_out

class Net(torch.nn.Module):
    def __init__(self, m, n, k, l, q) :
        super().__init__()
        self.blk1 = Residual_lin(300, 200, sp_l=100) # 480K
        self.blk2 = Residual_lin(300, 200, sp_l=100) # 480K
    def forward(self, X):
        print(X.size())
        s = torch.nn.Sequential(*[self.blk1, self.blk1])
        # add checkpoint over res_unit
        Y = checkpoint(s, X)
        return Y


class Net_ref(torch.nn.Module):
    def __init__(self, lw_1, lw_2):
        super().__init__()
        self.blk1 = Residual_lin_ref(300, 200)
        self.blk2 = Residual_lin_ref(300, 200)

        self.blk1.linear_1.weight = lw_1
        self.blk2.linear_1.weight = lw_1

        self.blk1.linear_2.weight = lw_2
        self.blk2.linear_2.weight = lw_2

    def forward(self, X):
        Y = self.blk1(X)
        Y = self.blk2(Y)
        return Y





def main():
    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = 400
    n = 300
    k = 200
    l = 700
    q = 1000

    # m = 40
    # n = 30
    # k = 20
    # l = 70
    # q = 100

    model = Net(m, n, k, l, q).to(device)
    memUsage = memory.MeasureMemory(device)

    print("==== model init ...")
    print(memUsage.snapshot())
    print(memUsage.currentValue())      #now should be around 480KB x 2
    print(memUsage.availableValue())
    
    input = torch.rand(m, k, requires_grad = True).cuda() # [400, 200] 320K
    print("==== alloc input tensor ...")
    print(memUsage.snapshot())
    print(memUsage.currentValue())      #now should be around +320K = 1.2M
    print(memUsage.availableValue())
    
    
    out = model(input)      # m x q

    # use same random as Net()
    l1_w = model.blk1.linear_1.weight
    l2_w = model.blk1.linear_2.weight
    # l3_w = model.linear_3.weight

    # print("l1_w", l1_w)
    # print("l2_w", l2_w)
    # print("l3_w", l3_w)


    model_ref = Net_ref(l1_w, l2_w).to(device)
    out_ref = model_ref(input)

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    # print("out ", out)
    # print("out_ref ", out_ref)
    
    print("~~ check forward correctness ~~")
    not_same_num = correctness_check.point_wise_compare_2d(m, k, out, out_ref)
    
    # #assert(torch.all(torch.eq(out, out_ref)))
    # #assert(torch.allclose(out, out_ref))

    

if __name__=="__main__":
    main()
