import torch
from uu.utils import memory 
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter



class Net(torch.nn.Module):
    def __init__(self, m, n, k, l, q) :
        super(Net, self).__init__()

        # suppose these 2 are streamable
        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l, bias=False) # weight: 840KB  300 700
        # last one is blocked
        self.linear_3 = torch.nn.Linear(l, q, bias=False) # weight:  2.8MB 700 1000

        self.stream = torch.nn.Sequential(*[self.linear_1, self.linear_2])

    def forward(self, x, sp_l):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)
        print("==== init ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be around 3.8MB
        print(memUsage.availableValue())

        temp = x[0:sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        out1 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        print("==== after 1st slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be + 280K
        print(memUsage.availableValue())

        temp = x[sp_l:2*sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        out2 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        print("==== after 2nd slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be + 280K
        print(memUsage.availableValue())

        temp = x[2*sp_l:3*sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        out3 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        print("==== after 3rd slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be + 280K
        print(memUsage.availableValue())


        temp = x[3*sp_l:4*sp_l, :]  
        tt = temp.to(device)  # 80K 100 200
        out4 = checkpoint(self.stream, tt)      #280K 100 700
        del tt

        print("==== after 4th slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be + 280K
        print(memUsage.availableValue())


        # now should be around 3.8+1.12 = 5M
        blocked_in = torch.cat([out1, out2, out3, out4], dim=0) #1.12M 400 700
        del out1
        del out2
        del out3
        del out4
        print("==== after concatenate slice ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should not increased
        print(memUsage.availableValue())

        out = self.linear_3(blocked_in) #1.6M 400 1000

        print("==== final ...")
        print(memUsage.snapshot())
        print(memUsage.currentValue())      #now should be +1.6, should around 6.5M
        print(memUsage.availableValue())



        return out


class Net_ref(torch.nn.Module):
    def __init__(self, m, n, k, l, q, l1_w, l2_w, l3_w) :
        super(Net_ref, self).__init__()

        self.linear_1 = torch.nn.Linear(k, n, bias=False) # weight: 240KB  200 300
        self.linear_2 = torch.nn.Linear(n, l, bias=False) # weight: 840KB  300 700
        self.linear_3 = torch.nn.Linear(l, q, bias=False) # weight:  2.8MB 700 1000

        self.linear_1.weight = l1_w
        self.linear_2.weight = l2_w
        self.linear_3.weight = l3_w
    
    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        out = self.linear_3(out)
        return out

def point_wise_compare(m, q, out, out_ref) -> int:
    count = 0
    for i in range(0, m):
        for j in range(0, q):
            if out[i,j] != out_ref[i, j]:
                if count <= 20:
                    print("out {:>25}, out_ref {:>25}, diff {:>25}".format(
                        out[i,j], out_ref[i,j], (out[i,j]-out_ref[i,j])
                    ))
                count += 1
    return count



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
    input = torch.rand(m, k) # [400, 200] 320K

    l1_w = model.linear_1.weight
    l2_w = model.linear_2.weight
    l3_w = model.linear_3.weight

    # print("l1_w", l1_w)
    # print("l2_w", l2_w)
    # print("l3_w", l3_w)

    # use same random as Net()
    model_ref = Net_ref(m, n, k, l, q, l1_w, l2_w, l3_w).to(device)
    input_ref = input.to(device)

    out = model(input, 100)      # m x q
    out_ref = model_ref(input_ref)
    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    # print("out ", out)
    # print("out_ref ", out_ref)
    
    # print("~~ check forward correctness ~~")
    # not_same_num = point_wise_compare(m, q, out, out_ref)
    # print (" precentile {} / {} = {}".format(not_same_num, m*q, (not_same_num/m/q)))
    #assert(torch.all(torch.eq(out, out_ref)))
    #assert(torch.allclose(out, out_ref))

    out.sum().backward()
    out_ref.sum().backward()

    # # manual compare grad for weight
    # assert(torch.all(torch.eq(model.linear_1.weight.grad, model_ref.linear_1.weight.grad)))
    # assert(torch.all(torch.eq(model.linear_2.weight.grad, model_ref.linear_2.weight.grad)))
    # assert(torch.all(torch.eq(model.linear_3.weight.grad, model_ref.linear_3.weight.grad)))


    out_w_list = []
    i = 0
    # for name, param in model.named_parameters():
    #     #print("i", i, name, param.size())
    #     i += 1
    #     out_w_list.append(param.grad)

    # out_ref_w_list = []
    # i = 0
    # for name, param in model_ref.named_parameters():
    #     #print("i", i, name, param.size(), param.grad.size())
    #     i += 1
    #     out_ref_w_list.append(param.grad)
       

    # print("~~ compare all weights ~~")
    # for ww in range(0, i):
    #     #print("out_w {} \n out_ref_w {}".format(out_w_list[ww], out_ref_w_list[ww]))
    #     assert(torch.all(torch.eq(out_w_list[ww], out_ref_w_list[ww])))




if __name__=="__main__":
    main()
