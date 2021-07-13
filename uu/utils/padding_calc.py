import torch
import torch.nn as nn
from typing import List

import pdb

def conv2d_padding_info(tile_coord: List, tile_size: List, prb_size: List, pads: List):
    #pdb.set_trace()
    tile_size_h = tile_size[0]
    tile_size_w = tile_size[1]
    H = prb_size[0]
    W = prb_size[1]

    tile_top = tile_coord[0]*tile_size_h
    tile_bottom = tile_top+tile_size_h-1
    tile_left = tile_coord[1]*tile_size_w
    tile_right = tile_left+tile_size_w-1


    pad_top = max(0-(tile_top-pads[0]), 0)
    pad_bottom = max((tile_bottom+pads[0])-(H-1), 0)

    pad_left = max(0-(tile_left-pads[1]), 0)
    pad_right = max((tile_right+pads[1])-(W-1), 0)

    #(\text{padding\_left}padding_left, \text{padding\_right}padding_right, \text{padding\_top}padding_top, \text{padding\_bottom}padding_bottom)
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]

    input_left = max(0, (tile_left-pads[1]))
    input_right = min(W-1, (tile_right+pads[1]))
    input_top = max(0, (tile_top-pads[0]))
    input_bottom = min(H-1, (tile_bottom+pads[0]))

    #input_tile view
    slice_info = [input_left, input_right, input_top, input_bottom]

    return padding_info, slice_info

def get_input_tile(output_tile_coord: List, H: int, W: int, Th: int, Tw: int, ph: int, pw: int, input):
    padding_info, slice_info = conv2d_padding_info(output_tile_coord, [Th, Tw], [H, W], [ph, pw])
    input_tile = input[:, :, slice_info[2]:slice_info[3]+1, slice_info[0]:slice_info[1]+1]       #NCHW
    
    # print(padding_info)
    # print(slice_info)

    pd = torch.nn.ConstantPad2d(padding_info, 0)
    input_tile = pd(input_tile)

    return input_tile





