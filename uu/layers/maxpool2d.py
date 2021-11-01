import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
import numpy as np
from torch.autograd.variable import Variable
import math
import pdb
from uu.utils import correctness_check 


class MMctx:
    def __init__(self):
        self.input = None
        self.kernel_size = None
        self.padding = None
        self.stride = None
        self.uniq_id = None
        self.info = None
        self.arg_max = None

myctx_dict = {}

class cMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^cMaxPool2dFunction fwd")
        input = inputs[0]
        kernel_size = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        #ctx.info = inputs[4]
        uniq_id = inputs[5]
        is_ccheckpoint = inputs[6]

        if uniq_id in myctx_dict.keys():
            #print("need to get existing")
            myctx = myctx_dict[uniq_id]
            # del myctx
            # myctx = MMctx()
        else:
            myctx = MMctx()

        if not is_ccheckpoint:
            out = F.max_pool2d(input, kernel_size, stride, padding, return_indices=True)
            out_value = out[0]
            out_index = out[1]
            # save status for bkward for non-checkpoint
            # ctx.stride = stride
            # ctx.kernel_size = kernel_size
            # ctx.padding = padding
            # ctx.info = inputs[4]
            # ctx.input = input
            # ctx.output = out_value
            # ctx.arg_max = out_index

            ctx.uniq_id = uniq_id
            myctx.stride = stride
            myctx.kernel_size = kernel_size
            myctx.padding = padding
            myctx.info = inputs[4]
            myctx.input = input
            myctx.arg_max = out_index
            del out
        else:
            out = F.max_pool2d(input, kernel_size, stride, padding, return_indices=True)
            out_value = out[0]
            del out
            #out_index = out[1]
        
        # place this entry
        myctx_dict[uniq_id] = myctx
        del input
        
        return out_value
    
    @staticmethod
    def backward(ctx, grad_output):
        # print("\n^^^^^cMaxPool2dFunction bwd")
        # #case1
        # if ctx.input.is_cuda:
        #     grad_in = maxpool_2d_bkw_cuda.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        # else:
        #     grad_in = maxpool_2d_bkw_cpp.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        
        myctx = myctx_dict[ctx.uniq_id]

        # print("input size", myctx.input.size())
        # print("grad_out size",grad_output.size())
        #print("grad_out ",grad_output)
        # print("arg size",myctx.arg_max.size())


        f_info = myctx.info[0][ctx.uniq_id]
        b_info = myctx.info[1][ctx.uniq_id]
        rev_g_depth = f_info.op_idex
        g_depth = b_info.op_idex    # global depth
        if g_depth == 0: 
            grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
            # reshape to tile size before leaving the segment

        elif rev_g_depth == 0:
            # the last stage in regular order
            new_grad_out = grad_output[:, :, b_info.input_slice[2]:b_info.input_slice[3]+1, b_info.input_slice[0]:b_info.input_slice[1]+1]
            #print("new_grad_out", new_grad_out.size())
            grad_in = torch._C._nn.max_pool2d_with_indices_backward(new_grad_out, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
        else:
            grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
    
        #print("##############grad_in in maxp", grad_in.size()) 
        #print("grad in", grad_in)
        torch.cuda.empty_cache()
        return grad_in, None, None, None, None, None, None
        
        
    # def backward(ctx, grad_output):
    #     #print("\n^^^^^cMaxPool2dFunction bwd")
    #     f_info = ctx.info[0][ctx.uniq_id]
    #     b_info = ctx.info[1][ctx.uniq_id]
    #     rev_g_depth = f_info.op_idex
    #     g_depth = b_info.op_idex    # global depth
    #     if g_depth == 0: 
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
    #         # reshape to tile size before leaving the segment

    #     elif rev_g_depth == 0:
    #         # the last stage in regular order
    #         new_grad_out = grad_output[:, :, b_info.input_slice[2]:b_info.input_slice[3]+1, b_info.input_slice[0]:b_info.input_slice[1]+1]
    #         #print("new_grad_out", new_grad_out.size())
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(new_grad_out, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
    #     else:
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        
        
    #     #print("##############grad_in in maxp", grad_in.size()) 
    #     #print("grad in", grad_in)
    #     return grad_in, None, None, None, None, None, None

class cMaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False, #mdepth = 1, num_maxp = 1
                 ):
        super(cMaxPool2d, self).__init__(kernel_size, stride,
                 padding, dilation, return_indices, ceil_mode)
        self.is_ccheckpoint = is_ccheckpoint

        # self.mdepth = mdepth # depth of a maxpool in the checkpoint segment
        # self.num_maxp = num_maxp

        

# do I need to create auto-fucntion for MaxPool??
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
        uniq_id = id(self)
        pi = info[0][uniq_id]
        
        if pi.op_idex == 0: # last stage in the segment or in the global network
            out = cmaxplool(input, self.kernel_size, self.stride,
                            self.padding, info, uniq_id, is_ccheckpoint)
            #print ("mxp FF", out.size())
            return out
        else:
            next_input = cmaxplool(input, self.kernel_size, self.stride,
                            self.padding, info, uniq_id, is_ccheckpoint)
            #print ("* mxp FF", next_input.size())
            return next_input, info, self.is_ccheckpoint



 
