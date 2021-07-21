import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from uu.utils import padding_calc


def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_output size : ', grad_output[0].size())
    print('ref grad_output  :\n ', grad_output[0])

    print('grad_input size : ', grad_input[0].size())
    print('ref grad_input  : \n', grad_input[0])
    



class Net_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        # self.conv2d_3 = nn.Conv2d(in_channels=1, 
        #                           out_channels=1, 
        #                           kernel_size=(3,3),
        #                           bias = False,
        #                           padding=(1,1)
        #                           )                                                    
        self.conv2d_1.weight = Parameter(torch.reshape(torch.arange(2, 20, step=2.0, dtype=torch.float), (1, 1, 3, 3)))
        self.conv2d_2.weight = Parameter(torch.reshape(torch.arange(3, 30, step=3.0, dtype=torch.float), (1, 1, 3, 3)))
        #self.conv2d_3.weight = w3

        # print(" w1 shape", self.conv2d_1.weight.size())
        # print(self.conv2d_1.weight)
        # print(" w2 shape", self.conv2d_2.weight.size())
        # print(self.conv2d_2.weight)

        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)

        self.act_1 = None #should be input
        self.act_2 = None #should be first conv output

    def forward(self, x):
        self.act_1 = x
        out = self.conv2d_1(x)
        self.act_2 = out
        #print("ref 1st out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        #out = self.conv2d_3(out)
        return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net_ref().to(device)
    input = torch.reshape(torch.arange(1, 10, step=1.0, dtype=torch.float), (1, 1, 3, 3)).cuda()
    input.requires_grad = True
    print("input shape", input.size())
    print(input)

    out = model(input)
    out.sum().backward()
    #print("\n***********\n")

    # print("ref conv2 weight grad\n", model.conv2d_2.weight.grad)
    # print("ref conv1 weight grad\n", model.conv2d_1.weight.grad)


    print("+++++++++++++++++++++++++++++++++++++++++++++++\n")
    #first g_out
    g_out = torch.ones(1, 1, 3, 3).cuda()
    print("last layer \"grad_inupt\" \n", g_out) 
    
    # tiled
    # print("\n------------------------------------------\n")
    conv2_t_bk = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  ).to("cuda")
    conv1_t_bk = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  ).to("cuda")      
    conv2_t_bk.weight = Parameter(torch.rot90(model.conv2d_2.weight.data, 2, [2,3]))
    conv1_t_bk.weight = Parameter(torch.rot90(model.conv2d_1.weight.data, 2, [2,3]))
    # print("ref conv2 weight \n", conv2_t_bk.weight)
    # print("ref conv1 weight \n", conv1_t_bk.weight)

    H = 3
    W = 3
    Th = 1
    Tw = 1
    num_conv = 2
    info = padding_calc.compute_info([0,0], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_1 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_1 = compute_grad_input(g_out_1, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_1\n", g_in_1)
    current_depth = 1
    g_in_1_prime = compute_grad_input(g_in_1, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_1_prime\n", g_in_1_prime)


    info = padding_calc.compute_info([0,1], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_2 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_2 = compute_grad_input(g_out_2, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_2\n", g_in_2)
    current_depth = 1
    g_in_2_prime = compute_grad_input(g_in_2, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_2_prime\n", g_in_2_prime)

    info = padding_calc.compute_info([0,2], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_3 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_3 = compute_grad_input(g_out_3, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_3\n", g_in_3)
    current_depth = 1
    g_in_3_prime = compute_grad_input(g_in_3, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_3_prime\n", g_in_3_prime)


    info = padding_calc.compute_info([1,0], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_4 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_4 = compute_grad_input(g_out_4, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_4\n", g_in_4)
    current_depth = 1
    g_in_4_prime = compute_grad_input(g_in_4, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_4_prime\n", g_in_4_prime)


    info = padding_calc.compute_info([1,1], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_5 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_5 = compute_grad_input(g_out_5, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_5\n", g_in_5)
    current_depth = 1
    g_in_5_prime = compute_grad_input(g_in_5, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_5_prime\n", g_in_5_prime)


    info = padding_calc.compute_info([1,2], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_6 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_6 = compute_grad_input(g_out_6, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_6\n", g_in_6)
    current_depth = 1
    g_in_6_prime = compute_grad_input(g_in_6, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_6_prime\n", g_in_6_prime)


    info = padding_calc.compute_info([2,0], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_7 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_7 = compute_grad_input(g_out_7, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_7\n", g_in_7)
    current_depth = 1
    g_in_7_prime = compute_grad_input(g_in_7, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_7_prime\n", g_in_7_prime)

    info = padding_calc.compute_info([2,1], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_8 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_8 = compute_grad_input(g_out_8, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_8\n", g_in_8)
    current_depth = 1
    g_in_8_prime = compute_grad_input(g_in_8, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_8_prime\n", g_in_8_prime)

    info = padding_calc.compute_info([2,2], H, W, Th, Tw, 1, 1, None, num_conv)
    current_depth = 0
    g_out_9 = padding_calc.get_input_tile(info, g_out, num_conv-current_depth-1)
    g_in_9 = compute_grad_input(g_out_9, num_conv-current_depth-1, info, conv2_t_bk)
    print("g_in_9\n", g_in_9)
    current_depth = 1
    g_in_9_prime = compute_grad_input(g_in_9, num_conv-current_depth-1, info, conv1_t_bk)
    print("g_in_9_prime\n", g_in_9_prime)
   

    out_row_1 = torch.cat([g_in_1_prime, g_in_2_prime, g_in_3_prime], dim=3)
    out_row_2 = torch.cat([g_in_4_prime, g_in_5_prime, g_in_6_prime], dim=3)
    out_row_3 = torch.cat([g_in_7_prime, g_in_8_prime, g_in_9_prime], dim=3)
    out = torch.cat([out_row_1, out_row_2, out_row_3], dim=2)

    print("final grad INPUT\n", out)



def compute_grad_input(grad_output, depth, info,func):
    if depth == 0:
        grad_input = func(grad_output)
    else:
        grad_input = func(grad_output)
        #print("net_ out\n", out)
        input_tile_for_next = padding_calc.recreate_input_tile(info, grad_input, depth-1)
        #print("shape input_tile_for_next\n", input_tile_for_next.size())
        #print("input_tile_for_next\n", input_tile_for_next)
        grad_input = input_tile_for_next
    return grad_input



if __name__=="__main__":
    main()
