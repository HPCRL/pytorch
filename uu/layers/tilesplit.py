import torch
from uu.utils import padding_calc
import numpy



class TiledSplitFunction(torch.autograd.Function):
    #big_grad_in = None
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
        b_info = ctx.info[1][-11]
        tile_coord = b_info.coord
        coord = b_info.input_slice
        big_grad_in = None
        # print(tile_coord)
        # print(ctx.num_tile)
        # print(coord[2], coord[3]+1, coord[0], coord[1]+1)
        print("TiledSplitFunction bwd", grad_output)
        # print(TiledSplitFunction.big_grad_in)
        if True or tile_coord == ctx.num_tile:
            # last one create the space
            if ctx.input_is_cuda:
                # create a cuda-tensor
                N = ctx.big_infput_shape[0]
                C = ctx.big_infput_shape[1]
                H = ctx.big_infput_shape[2]
                W = ctx.big_infput_shape[3]
                big_grad_in = torch.zeros(N, C, H, W).cuda()
            else:
                N = ctx.big_infput_shape[0]
                C = ctx.big_infput_shape[1]
                H = ctx.big_infput_shape[2]
                W = ctx.big_infput_shape[3]
                big_grad_in = torch.zeros(N, C, H, W) 
        
        
        
        big_grad_in[:,:, coord[2]:coord[3]+1, coord[0]:coord[1]+1] = grad_output
        if tile_coord == [0, 0]:
            print("TiledSplitFunction.big_grad_in", big_grad_in)
        
        
        return big_grad_in, None, None, None, None
       
class TiledSplit(torch.nn.Module):
    def __init__(self):
        super(TiledSplit, self).__init__()

    def forward(self, *inputs):
        #print("tsplit here")
        tsplit = TiledSplitFunction.apply
        r = tsplit(*inputs)
        return r
 
