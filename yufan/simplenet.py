import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import List

from graph import Graph, build_graph



class Net(nn.Module):
    def __init__(self, b=5, m=32, n=64, k=32) :
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(n,k)
        self.linear_2 = nn.Linear(k,16)

    def forward(self, x):


        result_1 = self.linear_1(x)
        result_2 = self.linear_2(result_1)
        #result_2 = checkpoint.checkpoint(self.linear_2, result_1)
        res = result_2 * 5
        return res




def dryrun_cpu(model: nn.Module, input_dim: List[int]) -> None:
    device_cpu = torch.device("cpu")
    model.to(device_cpu)
    input = torch.rand(input_dim).cpu()
    print("input tensor size", input.size())
    g = build_graph(model, input, True)
    #g.save("/uufs/chpc.utah.edu/common/home/u0940848/pytorch/yufan/test.pdf")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", default=5, type=int, help="Batch")
    parser.add_argument("--m", default=32, type=int, help="M Dimension")
    parser.add_argument("--n", default=64, type=int, help="N Dimension")
    parser.add_argument("--k", default=32, type=int, help="K Dimension")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs ")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--lr', type=float, default=0.01, help = "Learning rate")
    parser.add_argument('--gamma', type=float, default=0.7, help = "Learning rate decay ")
    parser.add_argument("--lr_step", type=int, default=1, help="Step LR Scheduler after this many epochs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = Net(b=args.b, m=args.m, n=args.n, k=args.k)
    
    #dryrun_cpu(model, [args.b, args.m, args.n])


    
    model.to(device)
    input = torch.rand(args.b, args.m, args.n).cuda()
    output = model(input)

    print("input size", input.size())
    print("output size", output.size())

    # g = make_dot( output, params=dict(model.named_parameters()))
    # g.view()

    # erro = output.sum()
    # erro.backward()

   
   
if __name__=="__main__":
    main()
