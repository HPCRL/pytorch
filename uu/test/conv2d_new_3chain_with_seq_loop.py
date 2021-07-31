from torch.autograd.grad_mode import no_grad
import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import conv2d, tilecat, sequential, tilesplit,tilecopy
from torch.nn.parameter import Parameter
#from torch.utils.checkpoint import checkpoint
from uu.utils import checkpoint


def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    # print('grad_output size : ', grad_output[0].size())
    # print('ref grad_output  :\n ', grad_output[0])

    # print('grad_input size : ', grad_input[0].size())
    # print('ref grad_input  : \n', grad_input[0])

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=16, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_3 = nn.Conv2d(in_channels=16, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )                                                    
        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_3.weight = Parameter(w3)
        self.relu = torch.nn.ReLU()

        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)
        self.conv2d_3.register_full_backward_hook(print_grad)

    def forward(self, x):
        out = self.conv2d_1(x)
        #print("ref 1st out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        out = self.conv2d_3(out)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: when we rewirte the network, we should know the depth info.
        # depth is 0 if it is the last conv2d, reversely increased
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=2,
                                  num_conv=3
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=16, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=1,
                                  num_conv=3
                                  )
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=16, 
                                  out_channels=16, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=0,
                                  num_conv=3
                                  )   
        self.tcat = tilecat.TiledConcatenateFunction.apply
        self.relu = torch.nn.ReLU()
        self.tcopy = tilecopy.TiledCopy()
        self.tsplit = tilesplit.TiledSplit()

        # TODO: How to make sequential work??                                                 
        #self.block = nn.Sequential(*[self.conv2d_1, self.conv2d_2])
        self.block = sequential.mSequential(*[self.conv2d_1, self.conv2d_2, self.conv2d_3])

    def forward(self, x, H, W, Th, Tw):
        num_conv = 3
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memUsage = memory.MeasureMemory(device)

        model_device = next(self.parameters()).is_cuda
        # TODO: here we have to somehow infer the shape of the output of the segment. 
        out = torch.zeros(1, 16, H, W, requires_grad=True).cuda()
        for i in range(0,4): 
            for j in range(0,4):
                coord = [i,j]
                info = padding_calc.compute_info(coord, H, W, Th, Tw, 1, 1, x, num_conv)
                input_tile = self.tsplit(info, x, num_conv-1, model_device)
                out_temp = self.block(input_tile, info)
                # default copy has issue in bkward
                # out[:,:, i*Th:(i+1)*Th, j*Tw:(j+1)*Tw] = out_temp
                
                # use customized copy
                tile_size = [Th, Tw]
                out = self.tcopy(out_temp, out, coord, tile_size)


        out = self.relu(out)
        return out


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    w3 = model.conv2d_3.weight.data

    #print(w1, w2)
    model_ref =  Net_ref(w1, w2, w3).to(device)
    #print(model_ref.conv2d_1.weight, model_ref.conv2d_2.weight)
    
    H = 16
    W = 16
    Th = int(H/4)
    Tw = int(W/4)
    input = torch.rand(1,3,H,W, requires_grad = True)
    # print("input shape", input.size())
    # print(input)
    input_ref = input.data
    input_ref = input_ref.cuda()


    out_ref = model_ref(input_ref)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, Th, Tw )
    

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")

    # print("out", out)
    # print("out_ref", out_ref)
    not_same_num = correctness_check.point_wise_compare_4d(1,16,H, W, out, out_ref)
    
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out_ref.sum().backward()
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out.sum().backward()
    

    # print("model.conv2d_1.weight.grad", model.conv2d_1.weight.grad)
    # print("model_ref.conv2d_1.weight.grad", model_ref.conv2d_1.weight.grad)
    # # print("model.conv2d_2.weight.grad", model.conv2d_2.weight.grad)
    # # print("model_ref.conv2d_2.weight.grad", model_ref.conv2d_2.weight.grad)
    # # print("model.conv2d_3.weight.grad", model.conv2d_3.weight.grad)
    # # print("model_ref.conv2d_3.weight.grad", model_ref.conv2d_3.weight.grad)
    assert(torch.allclose(model.conv2d_1.weight.grad, model_ref.conv2d_1.weight.grad, atol=1e-10))
    assert(torch.allclose(model.conv2d_2.weight.grad, model_ref.conv2d_2.weight.grad, atol=1e-10))
    assert(torch.allclose(model.conv2d_3.weight.grad, model_ref.conv2d_3.weight.grad, atol=1e-10))

    print("~~~~DONE TEST~~~~")





if __name__=="__main__":
    main()
