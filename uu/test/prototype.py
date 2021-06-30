import torch
from uu.utils import memory 

class Net(torch.nn.Module):
    def __init__(self, m, n, k, l) :

        #[m, k] * [k, n] --> [m, n]*[n, l] --> [m, l]
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(k, n) # weight: 240KB
        self.linear_2 = torch.nn.Linear(n, l) # weight: 840KB
        print("==== linear 1 weight shape", self.linear_1.weight.size())
        print("==== linear 2 weight shape", self.linear_2.weight.size())
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
       


    def forward(self, x):
        device = self.dummy_param.device
        print("device ", device)
        memUsage = memory.MeasureMemory(device)


        temp = torch.split(x, 200, dim=0) # 320K
        print("==== after split input, will double space?? ...") # looks not
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())
        
        result_1 = self.linear_1(temp[0])
        print("==== result_1 shape", result_1.size())
        print("==== after first Linear ...") # res1 [200*300] 240KB
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())


        result_2 = self.linear_1(temp[1])
        print("==== result_2 shape", result_2.size())
        print("==== after first Linear ...") # res2 [200*300] 240KB        
        print(memUsage.currentValue())
        print(memUsage.availableValue())
        print(memUsage.current())
        print(memUsage.available())
        print(memUsage.maxx())
        print(memUsage.currentCached())

        f_res1 = self.linear_2(result_1)
        f_res2 = self.linear_2(result_2)


        return [f_res1, f_res2]

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



    input = torch.rand(m, k).cuda() # 320K
    print("==== after allocate input ...")
    print(memUsage.currentValue())
    print(memUsage.availableValue())
    print(memUsage.current())
    print(memUsage.available())
    print(memUsage.maxx())
    print(memUsage.currentCached())

    model(input)


if __name__=="__main__":
    main()
