import torch.nn as nn

class mSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple or type(inputs) == list:
                #print("in customized Sequential", len(inputs))
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
