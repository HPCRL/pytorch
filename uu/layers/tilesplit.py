import torch
from uu.utils import padding_calc
import numpy

final_grad_out = None
class TiledSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        # here we have to decide sending to machine or not.
        # also we assume, the tiling only on H/W for a conv2d
        x = inputs[0] 
        info = inputs[1]
         
        first_op_in_seg = id(inputs[2])
        model_device = inputs[3]
        ctx.num_tile = inputs[4]
        ctx.input_is_cuda = x.is_cuda
        ctx.info = info
        ctx.big_infput_shape = x.size()
        
        input = padding_calc.get_input_tile(info[0], x, first_op_in_seg)
        if ctx.input_is_cuda != model_device:
            # print("#########", ctx.input_is_cuda)
            # print(model_device)
            if model_device == True: # model is on GPU 
                device = torch.device("cuda")
                input = input.to(device)    # explicitly load input tile to device 
            else:
                device = torch.device("cpu")
                input = input.to(device)    # explicitly load input tile to device 
        # print ("TiledSplitFunction input tile", input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: need to regather all tile-grad-in
        f_info = ctx.info[0][-11]
        b_info = ctx.info[1][-11]
        tile_coord = b_info.coord
        print(tile_coord)
        if tile_coord == ctx.num_tile:
            # last one create the space
            if ctx.input_is_cuda:
                # create a cuda-tensor
                N = ctx.big_infput_shape[0]
                C = ctx.big_infput_shape[1]
                H = ctx.big_infput_shape[2]
                W = ctx.big_infput_shape[3]
                grad_in = torch.zeros(N, C, H, W).cuda()
            else:
                N = ctx.big_infput_shape[0]
                C = ctx.big_infput_shape[1]
                H = ctx.big_infput_shape[2]
                W = ctx.big_infput_shape[3]
                grad_in = torch.zeros(N, C, H, W) 
        else:
            # copy in
            grad_in = torch.zeros(1,1,1,1) 

        return grad_in, None, None, None, None
       




class TiledSplit(torch.nn.Module):
    def __init__(self):
        super(TiledSplit, self).__init__()

    def forward(self, *inputs):
        #print("tsplit here")
        tsplit = TiledSplitFunction.apply
        r = tsplit(*inputs)
        return r
 
