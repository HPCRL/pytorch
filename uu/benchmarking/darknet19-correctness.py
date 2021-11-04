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
    # print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_output size : ', grad_output[0].size())
    # #print('ref grad_output  :\n ', grad_output[0])
    # print('grad_input size : ', grad_input[0].size())
   # print('ref grad_input  : \n', grad_input[0])
    li.append( grad_output[0])

def print_activa(self, input, output):
    # print('Inside '+ self.__class__.__name__+ ' forward')
    # print('input size : ', input[0].size())
    # print('output size : ', output[0].size())
    li_act.append(input[0])

    

Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 1

H = 512
W = 512
nTh = 4
nTw = 4
oH = H // 32
oW = W // 32

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=32, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #0
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))  #1

        self.conv2d_2 = nn.Conv2d(in_channels=32, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #2
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))  #3
                                
        
        self.conv2d_3 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #4
        self.conv2d_4 = nn.Conv2d(in_channels=128, 
                                  out_channels=64, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #5
        self.conv2d_5 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #6
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))  #7

        
        self.conv2d_6 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #8
        self.conv2d_7 = nn.Conv2d(in_channels=256, 
                                  out_channels=128, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #9
        self.conv2d_8 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)   
                                  )                 #10
        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))  #11

        
        self.conv2d_9 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #12
        self.conv2d_10 = nn.Conv2d(in_channels=512, 
                                  out_channels=256, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #13
        self.conv2d_11 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #14
        self.conv2d_12 = nn.Conv2d(in_channels=512, 
                                  out_channels=256, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #15
        self.conv2d_13 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #16
        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))  #17


        self.conv2d_14 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #18
        self.conv2d_15 = nn.Conv2d(in_channels=1024, 
                                  out_channels=512, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #19
        self.conv2d_16 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #20
        self.conv2d_17 = nn.Conv2d(in_channels=1024, 
                                  out_channels=512, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #21
        self.conv2d_18 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #22
        self.conv2d_19 = nn.Conv2d(in_channels=1024, 
                                  out_channels=1000,        ##fake one number
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #23


        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.sft = nn.Softmax(dim=-1)
        self.flat = nn.Flatten()
        

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
        self.conv2d_14.weight = Parameter(w14)
        self.conv2d_15.weight = Parameter(w15)
        self.conv2d_16.weight = Parameter(w16)
        self.conv2d_17.weight = Parameter(w17)
        self.conv2d_18.weight = Parameter(w18)
        self.conv2d_19.weight = Parameter(w19)
        
       
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
        self.conv2d_14.register_forward_hook(print_activa)
        self.conv2d_15.register_forward_hook(print_activa)
        self.conv2d_16.register_forward_hook(print_activa)
        self.conv2d_17.register_forward_hook(print_activa)
        self.conv2d_18.register_forward_hook(print_activa)
        self.conv2d_19.register_forward_hook(print_activa)
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
        self.conv2d_14.register_full_backward_hook(print_grad)
        self.conv2d_15.register_full_backward_hook(print_grad)
        self.conv2d_16.register_full_backward_hook(print_grad)
        self.conv2d_17.register_full_backward_hook(print_grad)
        self.conv2d_18.register_full_backward_hook(print_grad)
        self.conv2d_19.register_full_backward_hook(print_grad)
        self.maxpool1.register_full_backward_hook(print_grad)
        self.maxpool2.register_full_backward_hook(print_grad)
        self.maxpool3.register_full_backward_hook(print_grad)
        self.maxpool4.register_full_backward_hook(print_grad)
        self.maxpool5.register_full_backward_hook(print_grad)
        

        self.block = nn.Sequential(*[self.conv2d_1, self.maxpool1 ,self.conv2d_2, self.maxpool2, \
                                                self.conv2d_3,  self.conv2d_4, self.conv2d_5, self.maxpool3,  \
                                                self.conv2d_6, self.conv2d_7, self.conv2d_8, self.maxpool4, \
                                                self.conv2d_9, self.conv2d_10, self.conv2d_11, self.conv2d_12, self.conv2d_13, self.maxpool5, \
                                                self.conv2d_14, self.conv2d_15, self.conv2d_16, self.conv2d_17, self.conv2d_18, self.conv2d_19  ]) 

    def forward(self, x):
        out = self.block(x)
        print("out.shape", out.size())
        out = self.avgp(out)
        out = self.flat(out)
        print("out.shape", out.size())
        out = self.sft(out)
        print("out.shape", out.size())
        return out

#############################################


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3,
                                 out_channels=32,
                                 kernel_size=(Kh,Kw),
                                 bias = False,
                                 padding=(Ph,Pw)
                                 )                 #0
        self.maxpool1 = maxpool2d.cMaxPool2d((2,2), (2,2))  #1
    
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #2
        self.maxpool2 = maxpool2d.cMaxPool2d((2,2), (2,2))  #3
                                
        
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #4
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #5
        self.conv2d_5 = conv2d.TiledConv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #6
        self.maxpool3 = maxpool2d.cMaxPool2d((2,2), (2,2))  #7
    
        
        self.conv2d_6 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #8
        self.conv2d_7 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=128,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #9
        self.conv2d_8 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)  
                                    )                 #10
        self.maxpool4 = maxpool2d.cMaxPool2d((2,2), (2,2))  #11
    
        
        self.conv2d_9 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #12
        self.conv2d_10 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #13
        self.conv2d_11 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #14
        self.conv2d_12 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #15
        self.conv2d_13 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #16
        self.maxpool5 = maxpool2d.cMaxPool2d((2,2), (2,2))  #17
    
    
        self.conv2d_14 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #18
        self.conv2d_15 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #19
        self.conv2d_16 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #20
        self.conv2d_17 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #21
        self.conv2d_18 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #22
        self.conv2d_19 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=1000,        ##fake one number
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #23
        
        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.sft = nn.Softmax(dim=-1)
        self.flat = nn.Flatten()
        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.maxpool1 ,self.conv2d_2, self.maxpool2, \
                                                self.conv2d_3,  self.conv2d_4, self.conv2d_5, self.maxpool3,  \
                                                self.conv2d_6, self.conv2d_7, self.conv2d_8, self.maxpool4, \
                                                self.conv2d_9, self.conv2d_10, self.conv2d_11, self.conv2d_12, self.conv2d_13, self.maxpool5, \
                                                self.conv2d_14, self.conv2d_15, self.conv2d_16, self.conv2d_17, self.conv2d_18, self.conv2d_19]) #
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        #print("!!!!!!!", model_device)
        print("!!!!!!!", oH, oW)
        stream_structure = self.block1

    # prepare grad info for correctness check(only for linear )
        li_act_p = []
        for elm in li_act:
            pd = torch.nn.ConstantPad2d((Ph,Ph,Ph,Ph), 0)
            li_act_p.append(pd(elm))
         
        i = len(li)
        ii = 0

        #print("i", i, len(li_act))
        for op in self.block1._modules.values():
            if isinstance(op, tilesplit.TiledSplit):
               continue
            grad_dict_bk[id(op)*-1] = (li_act_p[ii], li[i-1])
            i -= 1
            ii+= 1
    # prepare grad info for correctness check(only for linear )


        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
                #print("info for", coord, info)
    # add grad_payload as negate keys
                info[0].update(grad_dict_bk)
      # add grad_payload as negate keys

                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("coord", coord)
                #out_temp = self.block1(x, info, stream_structure[1], model_device, [nTh, nTw])
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])
               
                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice

                # print(tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                
                #del info
        out = self.avgp(out)
        out = self.flat(out)
        out = self.sft(out)
        
        return out

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
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
    w14 = model.conv2d_14.weight.data
    w15 = model.conv2d_15.weight.data
    w16 = model.conv2d_16.weight.data
    w17 = model.conv2d_17.weight.data
    w18 = model.conv2d_18.weight.data
    w19 = model.conv2d_19.weight.data
    
    # add loss function here
    criterion =  nn.MSELoss()
    labels = torch.rand(batch, 1000).cuda()
    

    model_ref =  Net_ref(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19).to(device)
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    
    out_ref = model_ref(input_ref)
    print("done ref")
    print(out_ref.size())
    loss1 = criterion(out_ref, labels)
    print("loss1", loss1)
    loss1.backward()
    #out_ref.sum().backward()

  
    print("done ref bkw")


    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, nTh, nTw )

    

    

    #print(input_ref.grad)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    loss2 = criterion(out, labels)
    print("loss2", loss2)
    loss2.backward()

    #assert loss1 == loss2
    #out.sum().backward()

    print("~~ check forward correctness ~~")
    # print("out shape", out)
    # print("out_ref ", out_ref)
    # # not_same_num = correctness_check.point_wise_compare_4d(1,1,oH, oW, out, out_ref)
    correctness_check.check_equal(out, out_ref, False)

    # print("#### compare grad_in")
    # print("input ref grad", input_ref.grad[0,0,0,0])
    # print("input grad", input.grad[0,0,0,0])
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

    print("#### compare w14")
    correctness_check.check_equal(model_ref.conv2d_14.weight.grad, model.conv2d_14.weight.grad, False)

    print("#### compare w15")
    correctness_check.check_equal(model_ref.conv2d_15.weight.grad, model.conv2d_15.weight.grad, False)

    print("#### compare w16")
    correctness_check.check_equal(model_ref.conv2d_16.weight.grad, model.conv2d_16.weight.grad, False)

    print("#### compare w17")
    correctness_check.check_equal(model_ref.conv2d_17.weight.grad, model.conv2d_17.weight.grad, False)

    print("#### compare w18")
    correctness_check.check_equal(model_ref.conv2d_18.weight.grad, model.conv2d_18.weight.grad, False)

    print("#### compare w19")
    
    correctness_check.check_equal(model_ref.conv2d_19.weight.grad, model.conv2d_19.weight.grad, False)

    
if __name__=="__main__":
    main()