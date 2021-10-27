import torch
import torch.nn as nn
from torch.cuda import init
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import memory 
from uu.utils import checkpoint



Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 1
batch = 1
H = 8192
W = 8192
nTh = 4
nTw = 4

def func(input, memUsage):

    print("==== 1 padding ...")
    initmem = memUsage.currentValue()
    print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())     
    padding_info=[1,1,1,1]
    pd = torch.nn.ConstantPad2d(padding_info, 0)
    input = pd(input)

    print("==== 2 padding ...")
    initmem = memUsage.currentValue()
    print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())      

    return input




def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)


    input = torch.rand(batch,chanel,H,W, requires_grad = True).cuda()
    print("==== before padding ...")
    initmem = memUsage.currentValue()
    print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())     
    
    
    padding_info=[1,1,1,1]
    pd = torch.nn.ConstantPad2d(padding_info, 0)
    input = pd(input)

    #input = func(input, memUsage)


    print("==== after padding ...")
    initmem = memUsage.currentValue()
    print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())  

    print(input[0,0,0,1])   




if __name__=="__main__":
    main()