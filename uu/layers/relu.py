from torch.nn import functional as F
import torch.nn as nn

class cReLu(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super(cReLu, self).__init__()
        self.inplace = inplace

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
            print("missing info in cReLu")
            assert False

        next_input = F.relu(input, inplace=self.inplace)
        #print("in , out", input.size(), next_input.size())
        return next_input, info, self.is_ccheckpoint

