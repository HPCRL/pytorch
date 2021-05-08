# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import argparse

import torch
import torch.nn as nn

#from torch.utils.tensorboard import SummaryWriter
import graphviz
import torchviz as tv
from graphviz import Digraph

import torch.optim as optim
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import StepLR

import torch.autograd.profiler as profiler


# def make_dot(var, params):
#     """ Produces Graphviz representation of PyTorch autograd graph
    
#     Blue nodes are the Variables that require grad, orange are Tensors
#     saved for backward in torch.autograd.Function
    
#     Args:
#         var: output Variable
#         params: dict of (name, Variable) to add names to node that
#             require grad (TODO: make optional)
#     """
#     param_map = {id(v): k for k, v in params.items()}
#     print(param_map)
    
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()
    
#     def size_to_str(size):
#         return '('+(', ').join(['%d'% v for v in size])+')'

#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#     add_nodes(var.grad_fn)
#     return dot



class BranchLayerWithConcatenate(nn.Module):

    def __init__(self, b=5, m=32,n=64,k=32) :
        super(BranchLayerWithConcatenate, self).__init__()

        #self.T1_1 = nn.Parameter(torch.rand(b, m ,n), requires_grad=True)
        #self.T1_2 = nn.Parameter(torch.rand(b, n, k), requires_grad=True)
        self.input_1 = torch.rand(b,m,n)
        self.linear_1 = nn.Linear(n,k)

        #self.T2_1 = nn.Parameter(torch.rand(b, m, n), requires_grad=True)
        #self.T2_2 = nn.Parameter(torch.rand(b, n, k), requires_grad=True)
        self.input_2 = torch.rand(b,m,n)
        self.linear_2 = nn.Linear(n,k)

        #self.T3_1 = nn.Parameter(torch.rand(b, m, n), requires_grad=True)
        #self.T3_2 = nn.Parameter(torch.rand(b, n, k), requires_grad=True)
        self.input_3 = torch.rand(b,m,n)
        self.linear_3 = nn.Linear(n,k)

    def forward(self):

        #result_1 = torch.einsum('bmn, bnk->bmk', [self.T1_1, self.T1_2])
        #result_2 = torch.einsum('bmn, bnk->bmk', [self.T2_1, self.T2_2])
        #result_3 = torch.einsum('bmn, bnk->bmk', [self.T3_1, self.T3_2])

        result_1 = self.linear_1(self.input_1)
        result_2 = self.linear_2(self.input_2)
        result_3 = self.linear_3(self.input_3)

        final_result = torch.cat((result_1, result_2, result_3), 2)

        return final_result

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

    #set_seed(args.seed)


    model = BranchLayerWithConcatenate(b=args.b, m=args.m, n=args.n, k=args.k)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #print(list(model.parameters()))

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, verbose=False)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    default_input = torch.rand(args.b,args.m, 3*args.n)
    linear = nn.Linear(3*args.n, 3*args.k)

    output_default = linear(default_input)

    #writer = SummaryWriter()
    
    #print(model.state_dict)

    output = model()
    #g = make_dot(output, model.state_dict)

    #print(output)
    #print(dict(model.named_parameters()))

    # for v in output:
    #     #print(v)
    #     print(v.grad_fn)

    #g= tv.make_dot( output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    #g.view()

    #Training Part
    for ep in range(1, args.epochs + 1):
        model.train()

        # clears old gradients from the last step 
        # (otherwise you’d just accumulate 
        # the gradients from all loss.backward() calls).
        optimizer.zero_grad()
        output = model()
        loss = torch.norm(output - output_default, p='fro')
        #writer.add_scalar("Loss/train", loss, ep)
        #if ep % 50 == 0:
        print('For Epoch: {}, Loss:{}'.format(ep, loss.item()))

        #loss.backward() computes the derivative of the loss w.r.t. 
        # the parameters (or anything requiring gradients) using backpropagation.
        loss.backward(retain_graph=True)

        #opt.step() causes the optimizer to take a step based 
        # on the gradients of the parameters.

        optimizer.step()
        if ep % args.lr_step == 0:
            scheduler.step()

    writer.flush()

if __name__=="__main__":

    main()



