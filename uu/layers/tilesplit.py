import torch

class TiledSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        dim = input[-1]
        input = input[:-1]
        # print ("**input[0]", input[0])
        # print ("dim", dim)
        output = torch.cat(tensors=input, dim=dim, out=None)
        #output.requires_grad = True #tensors[0].requires_grad
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        #print("\n^^^^^grad_output", grad_output, grad_output.size())
        return grad_output, grad_output, grad_output, None


class TiledSplit(torch.nn.Module):
    def __init__(self):
        super(TiledSplit, self).__init__()

    def forward(self, inputs):
        #print("tsplit here")
        tsplit = TiledSplitFunction.apply
        r = tsplit(inputs)
        return r
 
