import torch
from uu.utils import padding_calc

class TiledSplitFunction(torch.autograd.Function):
    #big_grad_in = None
    @staticmethod
    def forward(ctx, *inputs):
        # here we have to decide sending to machine or not.
        # also we assume, the tiling only on H/W for a conv2d
        x = inputs[0] 
        info = inputs[1]
        #print("tsplit tile coor fwd", info[1][-11].coord)
         
        first_op_in_seg = id(inputs[2])
        model_device = inputs[3]
        ctx.num_tile = inputs[4]
        ctx.input_is_cuda = x.is_cuda
        ctx.info = info
        ctx.big_infput_shape = x.size()
        
        input = padding_calc.get_input_tile(info[0], x, first_op_in_seg)
        
    
        is_m_cuda = True if "cuda" in str(model_device) else False
        if ctx.input_is_cuda != is_m_cuda:
            # print("#########", ctx.input_is_cuda)
            # print(model_device)
            if is_m_cuda == True: # model is on GPU 
                #input = torch.cuda.FloatTensor(input.size())
                # print("TiledSplitFunction", input.size())
                input = input.to(model_device)    # explicitly load input tile to device 
            else:
                device = torch.device("cpu")
                input = input.to(device)    # explicitly load input tile to device 
        #print ("TiledSplitFunction input tile", input)
        return input
    
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: need to regather all tile-grad-in
        b_info = ctx.info[1][-11]
        coord = b_info.input_slice
        big_grad_in = None
        #print("tsplit tile coor bkw", tile_coord)
        # print(ctx.num_tile)
        
        if ctx.input_is_cuda:
            # create a cuda-tensor
            N = ctx.big_infput_shape[0]
            C = ctx.big_infput_shape[1]
            H = ctx.big_infput_shape[2]
            W = ctx.big_infput_shape[3]
            big_grad_in = torch.zeros(N, C, H, W).cuda()
            big_grad_in[:,:, coord[2]:coord[3]+1, coord[0]:coord[1]+1] = grad_output[:,:,0:H//ctx.num_tile[0], 0:W//ctx.num_tile[1]]
# if it a very begining input, we do not necessarily to pass it back.
        # else:
        #     N = ctx.big_infput_shape[0]
        #     C = ctx.big_infput_shape[1]
        #     H = ctx.big_infput_shape[2]
        #     W = ctx.big_infput_shape[3]
        #     big_grad_in = torch.zeros(N, C, H, W) 
        

        return big_grad_in, None, None, None, None
       
class TiledSplit(torch.nn.Module):
    def __init__(self):
        super(TiledSplit, self).__init__()

    def forward(self, *inputs):
        if len(inputs) == 5:
            is_ccheckpoint = False
        elif len(inputs) == 6:
            is_ccheckpoint = inputs[-1]
        else:
            print("missing info in split")
            assert False



        tsplit = TiledSplitFunction.apply
        r = tsplit(*inputs)
        info = inputs[1]
        return r, info, is_ccheckpoint
 
