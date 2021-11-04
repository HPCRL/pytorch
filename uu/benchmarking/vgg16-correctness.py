import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import checkpoint

li = []
li_act = []
grad_dict_bk = {}
def print_grad(self, grad_input, grad_output):
#     print('Inside '+ self.__class__.__name__+ ' backward')
#     print('grad_output size : ', grad_output[0].size())
#     #print('ref grad_output  :\n ', grad_output[0])
#     print('grad_input size : ', grad_input[0].size())
#    # print('ref grad_input  : \n', grad_input[0])
    li.append( grad_output[0])

def print_activa(self, input, output):
    # print('Inside '+ self.__class__.__name__+ ' forward')
    # print('input size : ', input[0].size())
    # #print('input : ', input[0])
    # print('output size : ', output[0].size())
    li_act.append(input[0])

    

Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 2

H = 512
W = 512
oH = H//32
oW = W//32
nTh = 1
nTw = 1

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, fcw1, fcw2):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=64, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_3 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_4 = nn.Conv2d(in_channels=128, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))                          
        self.conv2d_5 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_6 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_7 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_8 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_9 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_10 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_11 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_12 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_13 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))
        self.relu = nn.ReLU()
        
        
        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.flat = nn.Flatten()
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=-1)
        
        
        # self.flat = nn.Flatten()
        # in_feature = chanel*oH*oW
        # self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, 1024, bias=False)



        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_3.weight = Parameter(w3)
        self.conv2d_4.weight = Parameter(w4)
        self.conv2d_5.weight = Parameter(w5)
        self.conv2d_6.weight = Parameter(w6)
        self.conv2d_7.weight = Parameter(w7)
        self.conv2d_8.weight = Parameter(w8)
        self.conv2d_9.weight = Parameter(w9)
        self.conv2d_10.weight = Parameter(w10)
        self.conv2d_11.weight = Parameter(w11)
        self.conv2d_12.weight = Parameter(w12)
        self.conv2d_13.weight = Parameter(w13)
        self.fc1.weight = Parameter(fcw1)
        #self.fc2.weight = Parameter(fcw2)
       
        self.conv2d_1.register_forward_hook(print_activa)
        self.conv2d_3.register_forward_hook(print_activa)
        self.conv2d_4.register_forward_hook(print_activa)
        self.conv2d_2.register_forward_hook(print_activa)
        self.conv2d_5.register_forward_hook(print_activa)
        self.conv2d_6.register_forward_hook(print_activa)
        self.conv2d_7.register_forward_hook(print_activa)
        self.conv2d_8.register_forward_hook(print_activa)
        self.conv2d_9.register_forward_hook(print_activa)
        self.conv2d_10.register_forward_hook(print_activa)
        self.conv2d_11.register_forward_hook(print_activa)
        self.conv2d_12.register_forward_hook(print_activa)
        self.conv2d_13.register_forward_hook(print_activa)
        self.maxpool1.register_forward_hook(print_activa)
        self.maxpool2.register_forward_hook(print_activa)
        self.maxpool3.register_forward_hook(print_activa)
        self.maxpool4.register_forward_hook(print_activa)
        self.maxpool5.register_forward_hook(print_activa)
        
        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)
        self.conv2d_3.register_full_backward_hook(print_grad)
        self.conv2d_4.register_full_backward_hook(print_grad)
        self.conv2d_5.register_full_backward_hook(print_grad)
        self.conv2d_6.register_full_backward_hook(print_grad)
        self.conv2d_7.register_full_backward_hook(print_grad)
        self.conv2d_8.register_full_backward_hook(print_grad)
        self.conv2d_9.register_full_backward_hook(print_grad)
        self.conv2d_10.register_full_backward_hook(print_grad)
        self.conv2d_11.register_full_backward_hook(print_grad)
        self.conv2d_12.register_full_backward_hook(print_grad)
        self.conv2d_13.register_full_backward_hook(print_grad)
        self.maxpool1.register_full_backward_hook(print_grad)
        self.maxpool2.register_full_backward_hook(print_grad)
        self.maxpool3.register_full_backward_hook(print_grad)
        self.maxpool4.register_full_backward_hook(print_grad)
        self.maxpool5.register_full_backward_hook(print_grad)

        self.block1 = nn.Sequential(*[self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.maxpool1, \
                                               self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.maxpool2,  \
                                               self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.maxpool3, \
                                               self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.maxpool4, \
                                               self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.maxpool5, \
                                             ]) 

    def forward(self, x):
        out = self.block1(x)
        out = self.avgp(out)

        print("out.shape", out.size())
        out = self.flat(out)
        print("out.shape", out.size())
        out = self.fc1(out)
        print("out.shape", out.size())
        out = self.sft(out)
        print("out.shape", out.size())

    
        return out
        





class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=64, 
                                        out_channels=64, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_3 = conv2d.TiledConv2d(in_channels=64, 
                                        out_channels=128, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=128, 
                                        out_channels=128, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp2 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_5 = conv2d.TiledConv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_6 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=256, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.conv2d_7 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=256, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.mxp3 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_8 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_9 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_10 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp4 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_11 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_12 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_13 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp5 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.relu = relu.cReLu()
        # in_feature = chanel*oH*oW
        # self.flat = nn.Flatten()
        # self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, 1024, bias=False)
        


        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.flat = nn.Flatten()
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=-1)
        

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.mxp1, \
                                                self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.mxp2,  \
                                                self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.mxp3, \
                                                self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.mxp4, \
                                                self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.mxp5   ]) #
        
        # self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.conv2d_2, self.mxp1, \
        #                                         self.conv2d_3,  self.conv2d_4,  self.mxp2,  \
        #                                         self.conv2d_5,  self.conv2d_6, self.conv2d_7,  self.mxp3, \
        #                                         self.conv2d_8,  self.conv2d_9, self.conv2d_10,  self.mxp4, \
        #                                         self.conv2d_11,  self.conv2d_12,  self.conv2d_13,  self.mxp5,   ]) #
        

    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        # print("!!!!!!!", len(shape_dict))
        # print("!!!!!!!", oH, oW)
        stream_structure = self.block1

    # prepare grad info for correctness check(only for linear )
        li_act_p = []
        for elm in li_act:
            pd = torch.nn.ConstantPad2d((Ph,Ph,Ph,Ph), 0)
            li_act_p.append(pd(elm))
        i = len(li)
        ii = 0
        for op in self.block1._modules.values():
            if isinstance(op, tilesplit.TiledSplit) or isinstance(op, relu.cReLu):
               continue
            grad_dict_bk[id(op)*-1] = (li_act_p[ii], li[i-1])
            i -= 1
            ii+= 1
    # prepare grad info for correctness check(only for linear )


        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                
                # TODO: here we have to somehow provide static info and num_conv. 
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
    # add grad_payload as negate keys
                #info[0].update(grad_dict_bk)
      # add grad_payload as negate keys
                
                # print("++++++++++++++++++++++++++++++++++++++++++++++++")
                # print("coord", coord)
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])
                
                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice
                #print("FF", tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                
                #del info
        
        out = self.avgp(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.sft(out)

        return out


import time

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    
    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    w3 = model.conv2d_3.weight.data
    w4 = model.conv2d_4.weight.data
    w5 = model.conv2d_5.weight.data
    w6 = model.conv2d_6.weight.data
    w7 = model.conv2d_7.weight.data
    w8 = model.conv2d_8.weight.data
    w9 = model.conv2d_9.weight.data
    w10 = model.conv2d_10.weight.data
    w11 = model.conv2d_11.weight.data
    w12 = model.conv2d_12.weight.data
    w13 = model.conv2d_13.weight.data

    fcw1 = model.fc1.weight.data
    # fcw2 = model.fc2.weight.data
    fcw2= None
    
    model_ref =  Net_ref(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, fcw1, fcw2).to(device)
    
    start_time = time.time()
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    ref_elapsed_fwd = time.time() - start_time

    out_ref.sum().backward()
    
    ref_elapsed_total = time.time() - start_time
    print("done ref")
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n", ref_elapsed_fwd, ref_elapsed_total)
    
    
    start_time = time.time()
    out = model(input, H, W, nTh, nTw )
    our_elapsed_fwd = time.time() - start_time
   
    out.sum().backward()
    our_elapsed_total = time.time() - start_time

    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n", our_elapsed_fwd, our_elapsed_total)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("~~ check forward correctness ~~")
    # print("out shape", out)
    # print("out_ref ", out_ref)
    # # not_same_num = correctness_check.point_wise_compare_4d(1,1,oH, oW, out, out_ref)
    correctness_check.check_equal(out, out_ref, False)

    # print("#### compare grad_in")
    # # print("input ref grad", input_ref.grad)
    # # print("input grad", input.grad)
    # #not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, input_ref.grad.to('cpu'))
    # correctness_check.check_equal(input.grad, input_ref.grad, False)


    print("#### compare w1")
    correctness_check.check_equal(model_ref.conv2d_1.weight.grad, model.conv2d_1.weight.grad, False)

    print("#### compare w2")
    correctness_check.check_equal(model_ref.conv2d_2.weight.grad, model.conv2d_2.weight.grad, False)

    print("#### compare w3")
    correctness_check.check_equal(model_ref.conv2d_3.weight.grad, model.conv2d_3.weight.grad, False)

    print("#### compare w4")
    correctness_check.check_equal(model_ref.conv2d_4.weight.grad, model.conv2d_4.weight.grad, False)

    print("#### compare w5")
    correctness_check.check_equal(model_ref.conv2d_5.weight.grad, model.conv2d_5.weight.grad, False)

    print("#### compare w6")
    correctness_check.check_equal(model_ref.conv2d_6.weight.grad, model.conv2d_6.weight.grad, False)

    print("#### compare w7")
    correctness_check.check_equal(model_ref.conv2d_7.weight.grad, model.conv2d_7.weight.grad, False)


    print("#### compare w8")
    correctness_check.check_equal(model_ref.conv2d_8.weight.grad, model.conv2d_8.weight.grad, False)


    print("#### compare w9")
    correctness_check.check_equal(model_ref.conv2d_9.weight.grad, model.conv2d_9.weight.grad, False)


    print("#### compare w10")
    correctness_check.check_equal(model_ref.conv2d_10.weight.grad, model.conv2d_10.weight.grad, False)

    print("#### compare w11")
    correctness_check.check_equal(model_ref.conv2d_11.weight.grad, model.conv2d_11.weight.grad, False)

    print("#### compare w12")
    correctness_check.check_equal(model_ref.conv2d_12.weight.grad, model.conv2d_12.weight.grad, False)

    print("#### compare w13")
    correctness_check.check_equal(model_ref.conv2d_13.weight.grad, model.conv2d_13.weight.grad, False)


if __name__=="__main__":
    main()