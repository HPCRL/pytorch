import torch
from uu.utils import padding_calc
import numpy

final_grad_out = None
class TiledSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        # here we have to decide sending to machine or not.
        # also we assume, the tiling only on H/W for a conv2d
        info = inputs[0]
        x = inputs[1]  
        depth = inputs[2]
        model_device = inputs[3]
        
        ctx.input_is_cuda = x.is_cuda

        ctx.abs_id = numpy.prod(info[0].coord)  
        ctx.big_infput_shape = x.size()
        ctx.coord = info[0].coord
        input = padding_calc.get_input_tile(info, x, depth)
        if ctx.input_is_cuda != model_device:
            # print(ctx.input_is_cuda)
            # print(model_device)
            if model_device == True: # model is on GPU 
                device = torch.device("cuda")
                input = input.to(device)    # explicitly load input tile to device 
            else:
                device = torch.device("cpu")
                input = input.to(device)    # explicitly load input tile to device 
        #print ("TiledSplitFunction input tile", input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None
        # if ctx.abs_id == 0: 
        #     # very first tile
        #     return
        # else:
        #     final_grad_out = torch.zeros(ctx.big_infput_shape)




class TiledSplit(torch.nn.Module):
    def __init__(self):
        super(TiledSplit, self).__init__()

    def forward(self, *inputs):
        #print("tsplit here")
        tsplit = TiledSplitFunction.apply
        r = tsplit(*inputs)
        return r
 
