import torch

class TiledConcatenateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        dim = inputs[-1]
        input = inputs[:-1]
        ctx.input_num = len(input)
        # print ("**input[0]", input[0])
        # print ("dim", dim)
        output = torch.cat(tensors=input, dim=dim, out=None)
        #output.requires_grad = True #tensors[0].requires_grad
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # print("\n^^^^^grad_output", grad_output, grad_output.size())
        # based on num of input to generate return tuple
        res = list()
        for i in range(0,ctx.input_num):
            res.append(grad_output)
        res.append(None)    # last arg is dim, no need for grad
        res = tuple(res)

        return res

class TiledCat(torch.nn.Module):
    def __init__(self):
        super(TiledCat, self).__init__()

    def forward(self, *inputs):
        tcat = TiledConcatenateFunction.apply
        r = tcat(*inputs)
        return r
 
