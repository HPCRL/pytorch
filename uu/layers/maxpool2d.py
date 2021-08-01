import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t



# class cMaxPool2dFunction(torch.autograd.Function):
#     # create a static variable
#     @staticmethod
#     def forward(ctx, *inputs):
#         #print("\n^^^^^TiledCopyFunction fwd")
#         out_temp = inputs[0]
#         out = inputs[1]
#         coord = inputs[2]
#         tile_size = inputs[3]
       
#         #ctx.input_num = len(inputs)
#         # print ("**input[0]", input[0])
#         # print ("dim", dim)
#         F.max_pool2d(input, kernel_size, stride,
#                             self.padding, dilation, ceil_mode,
#                             self.return_indices)

#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         #print("\n^^^^^TiledCopyFunction")
        
#         # based on num of input to generate return tuple
#         res = list()
#         res.append(None)    # last arg is dim, no need for grad
#         res.append(None)
#         res.append(None)
#         res = tuple(res)


#         # print(TiledCopyFunction.GRAD_OUT)
#         return res



class cMaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = None,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False, is_last = False
                 ):
        super(cMaxPool2d, self).__init__(kernel_size, stride,
                 padding, dilation, return_indices, ceil_mode)
        self.is_ccheckpoint = is_ccheckpoint
        self.is_last = is_last

        print(is_ccheckpoint, is_last)


    def forward(self, *inputs):
        if type (inputs[0]) == tuple:
            # to remove additional packing in tuple
            inputs = list(inputs[0])

        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        elif len(inputs) == 3:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        else:
            print("missing info in cMaxPool2d")
            assert False
        #cmaxplool = cMaxPool2dFunction.apply
        # TODO: checkpoint ???
        if self.is_last:
            return F.max_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
        else:
            next_input = F.max_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
            # need to handle padded for next if needed. 
            
            return next_input, info, self.is_ccheckpoint
        
        
 
