import torch


class TiledCopyFunction(torch.autograd.Function):
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
        print ("coord", coord)
        print("out_temp", out_temp.size())
        out[:,:, coord[2]:coord[3]+1, coord[0]:coord[1]+1] = out_temp
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



class TiledCopy(torch.nn.Module):
    def __init__(self):
        super(TiledCopy, self).__init__()

    def forward(self, *inputs):
        tcopy = TiledCopyFunction.apply
        r = tcopy(*inputs)
        return r
 
