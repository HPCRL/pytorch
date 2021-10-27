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
H = 4096
W = 4096
nTh = 4
nTw = 4

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        
        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        
        
       

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        out = self.maxpool1(out)
        

        return out




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.conv2d_2, self.mxp1, \
                                                
                                                ]) #
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        print("!!!!!!!", N, C, oH, oW)
        stream_structure = self.block1
        memUsage = memory.MeasureMemory(model_device)


        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        print("==== after alloc final out ...")
        initmem = memUsage.currentValue()
        print(memory.MemSize(initmem))   



        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("coord", coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)

                print("==== after calc info ...", coord)
                initmem = memUsage.currentValue()
                print(memory.MemSize(initmem))   

                #chkpoint
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])

                # input_tile = self.tsplit(x, info, stream_structure[1], model_device, [nTh, nTw]) # -1 here is to match 0-base
                # out_temp = self.block1(x, info, stream_structure[1], model_device, [nTh, nTw])
                print("==== ** before copy ...", coord)
                initmem = memUsage.currentValue()
                print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())    


                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice

                out = self.tcopy(out_temp, out, output_index, tile_size)
                del out_temp
                
                print("==== loop ...", coord)
                initmem = memUsage.currentValue()
                print(initmem, memory.MemSize(initmem),  memUsage.maximumValue(), memUsage.maxx())     
                del info

        return out

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)


    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # w1 = model.conv2d_1.weight.data
    # w2 = model.conv2d_2.weight.data
   

    memUsage = memory.MeasureMemory(device)

    print("==== init ...")
    initmem = memUsage.currentValue()
    print(memory.MemSize(initmem))      #now should be around 3.8MB
    print(memUsage.available())



    # # model_ref =  Net_ref(w1, w2).to(device)
    # # input_ref = input.data.clone() 
    # # input_ref = input_ref.cuda()
    # # input_ref.requires_grad = True
    # # out_ref = model_ref(input_ref)

   
    # # print("==== ref_fwd done ...")
    # # ref_fwd_use = memUsage.currentValue()-initmem
    # # # print(memory.MemSize(ref_fwd_use) )     #now should be around 3.8MB
    # # # print(memUsage.available())
    # # print("max ref", memUsage.maximumValue())



    # # print("done ref")
    # # out_ref.sum().backward()
    # # print("done ref bkw")

    # # print("==== ref_bwd done ...")
    # # # ref_bwd_use = memUsage.currentValue()-ref_fwd_use
    # # # ref_bwd_use_total = memUsage.currentValue()-initmem
    # # # print("ref_bwd_use",memory.MemSize(ref_bwd_use))      
    # # # print("ref_bwd_use t", memory.MemSize(ref_bwd_use_total))     
    # # # print(memUsage.available())
    # # print("max ref", memUsage.maximumValue())



    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")

    print("==== our init ...")
    our_initmem = memUsage.currentValue()
    print(memory.MemSize(our_initmem))      #now should be around 3.8MB
    #print(memUsage.available())
    out = model(input, H, W, nTh, nTw )

    print("==== our_fwd done ...")
    our_fwd_use = memUsage.currentValue()-our_initmem
    print(memory.MemSize(our_fwd_use) )     #now should be around 3.8MB
    #print(memUsage.available())
    print("max fwd", memUsage.maximumValue(), memUsage.maxx())

   
    #print(input_ref.grad)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out.sum().backward()
    del out
    print("==== our_bwd done ...")
    our_bwd_use = memUsage.currentValue()-our_fwd_use
    our_bwd_use_total = memUsage.currentValue()-our_initmem
    print("our_bwd_use", memory.MemSize(our_bwd_use))   
    print("our_bwd_use t", memory.MemSize(our_bwd_use_total))    
    print(memUsage.available())
    print("max our", memUsage.maximumValue(), memUsage.maxx())





    #
    # print("==== before padding ...")
    # our_initmem = memUsage.currentValue()
    # print(memory.MemSize(our_initmem))     
    # padding_info=[1,1,1,1]
    # pd = torch.nn.ConstantPad2d(padding_info, 0)
    # input = pd(input)

    # print("==== after padding ...")
    # our_initmem = memUsage.currentValue()
    # print(memory.MemSize(our_initmem))  

    # print(input[0,0,0,1])   




if __name__=="__main__":
    main()