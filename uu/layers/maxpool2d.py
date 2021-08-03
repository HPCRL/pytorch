import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
import numpy as np
from torch.autograd.variable import Variable


class cMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        print("\n^^^^^cMaxPool2dFunction fwd")
        input = inputs[0]
        
        kernel_size = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        
        
        in_height = input.size()[2]
        in_width = input.size()[3]

        w_height = kernel_size[1]
        w_width = kernel_size[0]

        h_pad = padding[1]
        w_pad = padding[0]

        h_stride = stride[0]
        w_stride = stride[1]

        out_height = int((in_height - w_height + 2 * h_pad) / h_stride) + 1
        out_width = int((in_width - w_width + 2 * w_pad) / w_stride) + 1

        B = input.size()[0]
        C = input.size()[1]

        out = torch.zeros((B, C, out_height, out_width)).cuda()
        # print(out.size())
        # print(in_height)
        # print(w_height)
        # print(h_pad)
        # print(h_stride)

        arg_max = np.zeros((B, C, out_height, out_width), dtype=np.int32)
        for b in range(B):
            for c in range(C):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * h_stride
                        start_j = j * w_stride
                        end_i = start_i + w_height
                        end_j = start_j + w_width
                        out[b, c, i, j] = torch.max(input[:, :, start_i: end_i, start_j: end_j])
                        arg_max[b, c, i, j] = torch.argmax(input[:, :, start_i: end_i, start_j: end_j])

        ctx.arg_max = arg_max
        ctx.stride = stride
        ctx.out_height = out_height
        ctx.out_width = out_width
        ctx.kernel_size = kernel_size
        ctx.B = B
        ctx.C = C

        ctx.input = input
        # print(out.size())
        return Variable(out)
    
    @staticmethod
    def backward(ctx, grad_output):
        print("\n^^^^^cMaxPool2dFunction bwd")
        print(grad_output.size())
        print(ctx.input)
        print(ctx.arg_max)

        w_height = ctx.kernel_size[1]
        w_width = ctx.kernel_size[0]

        grad_in = torch.zeros((ctx.B, ctx.C, grad_output.size()[2]*2,grad_output.size()[3]*2)).cuda()
        print(grad_in.size())
        for b in range(ctx.B):
            for c in range(ctx.C):
                for i in range(grad_output.size()[2]):
                    for j in range(grad_output.size()[3]):
                        start_i = i * ctx.stride[1]
                        start_j = j * ctx.stride[0]
                        index = np.unravel_index(ctx.arg_max[b, c, i, j], ctx.kernel_size)
                        print(index)
                        print(start_i+index[0], start_j+index[1])
                        grad_in[b, c, start_i+index[0], start_j+index[1]] = grad_output[b, c, i, j]
                        

        return Variable(grad_in), None, None, None





class cMaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
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
        cmaxplool = cMaxPool2dFunction.apply
        # TODO: checkpoint ???

        if self.is_last:
            return cmaxplool(input, self.kernel_size, self.stride,
                                self.padding)
        else:
            next_input = cmaxplool(input, self.kernel_size, self.stride,
                                self.padding)
            # need to handle padded for next if needed. 
            return next_input, info, self.is_ccheckpoint
        
        
 
