from  torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t


class cBatchNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, is_ccheckpoint=False
                 ):
        super(cBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.is_ccheckpoint = is_ccheckpoint


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
        
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        uniq_id = id(self)
        #pi = info[0][uniq_id]
        
        # if pi.op_idex == 0: # last stage in the segment or in the global network
        #     next_input = F.batch_norm(input)
        #     return next_input
        # else:
        #     next_input = F.batch_norm(input)
        #     return next_input, info, self.is_ccheckpoint
        
        # TODO: 'running_mean' and 'running_var'??
        next_input = F.batch_norm(input, None, None)
        return next_input, info, self.is_ccheckpoint
