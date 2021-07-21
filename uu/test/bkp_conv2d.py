import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_output size : ', grad_output[0].size())
    print('grad_output  :\n ', grad_output[0])

    print('grad_input size : ', grad_input[0].size())
    print('grad_input  : \n', grad_input[0])
    



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

        print("w1 shape", self.conv2d_1.weight.size())
        print(self.conv2d_1.weight)
        print("w2 shape", self.conv2d_2.weight.size())
        print(self.conv2d_2.weight)

        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)

    def forward(self, x):
        out = self.conv2d_1(x)
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

    print("+++++++++++++++++++++++++++++++++++++++++++++++\n")
    #first g_out
    g_out = torch.ones(1, 1, 3, 3).cuda()
    print("last layer \"grad_inupt\" \n", g_out) 
    full_conv2 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(2,2)
                                  ).to("cuda")
    full_conv1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(2,2)
                                  ).to("cuda")                              
    
    # rotate weight                              
    full_conv2.weight = Parameter(torch.rot90(model.conv2d_2.weight.data, 2, [2,3]))
    #print(full_conv2.weight)
    g_in = full_conv2(g_out)
    print("g_in\n", g_in)
    # remove padded
    g_out_prime = g_in[:,:, 1:4, 1:4] 
    print("after crop \n", g_out_prime)


    # rotate weight                              
    full_conv1.weight = Parameter(torch.rot90(model.conv2d_1.weight.data, 2, [2,3]))
    #print(full_conv1.weight)
    g_in_prime = full_conv1(g_out_prime)
    print("g_in\n", g_in_prime)

    g_out_prime_prime = g_in_prime[:,:, 1:4, 1:4] 
    print("after crop \n", g_out_prime_prime)



if __name__=="__main__":
    main()
