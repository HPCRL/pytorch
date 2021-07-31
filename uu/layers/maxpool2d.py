import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t



class cMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    GRAD_OUT = None
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^TiledCopyFunction fwd")
        out_temp = inputs[0]
        out = inputs[1]
        coord = inputs[2]
        tile_size = inputs[3]
       
        #ctx.input_num = len(inputs)
        # print ("**input[0]", input[0])
        # print ("dim", dim)
        out[:,:, coord[0]*tile_size[0]:(coord[0]+1)*tile_size[0], coord[1]*tile_size[1]:(coord[1]+1)*tile_size[1]] = out_temp
        #output.requires_grad = True #tensors[0].requires_grad
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        #print("\n^^^^^TiledCopyFunction")
        if TiledCopyFunction.GRAD_OUT is None:
            TiledCopyFunction.GRAD_OUT = grad_output

        # based on num of input to generate return tuple
        res = list()
        res.append(TiledCopyFunction.GRAD_OUT)
        res.append(None)    # last arg is dim, no need for grad
        res.append(None)
        res.append(None)
        res = tuple(res)


        # print(TiledCopyFunction.GRAD_OUT)
        return res



class cMaxPool2d(_MaxPoolNd):
    def __init__(self):
        super(cMaxPool2d, self).__init__()

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, *inputs):
        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        else:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        cmaxplool = cMaxPool2dFunction.apply

        if self.depth == 0:
            return cmaxplool(input, info, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices, self.is_ccheckpoint)
        else:
            return cmaxplool(input, info, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices, self.is_ccheckpoint), info, self.is_ccheckpoint
 
