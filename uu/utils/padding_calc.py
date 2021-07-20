import torch
import torch.nn as nn
from typing import Dict, List

import pdb

def conv2d_padding_info(tile_indx: List, prb_size: List, pads: List):
    #pdb.set_trace()
    H = prb_size[0]
    W = prb_size[1]

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]


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

    iexp_top = pads[0] if (tile_top-pads[0])>=0 else 0
    iexp_bottom = pads[0] if (tile_bottom+pads[0])<=(H-1) else 0
    iexp_left = pads[1] if (tile_left-pads[1]) >= 0 else 0
    iexp_right = pads[1] if (tile_right+pads[1])<=(W-1) else 0
    internal_expand = [iexp_left, iexp_right, iexp_top, iexp_bottom]

    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]
    
    return padding_info, slice_info, internal_expand, real_index

# might need to create a memo structure. 
class Pad_info:
    def __init__(self, padding_info, slice_info, internal_expand, real_index):
        self.padding_info = padding_info
        self.slice_info = slice_info
        self.internal_expand = internal_expand
        self.real_index = real_index

def get_input_tile(output_tile_coord: List, H: int, W: int, Th: int, Tw: int, ph: int, pw: int, input, depth_convs: int) -> Dict:
    info_dict = {}
    tile_top = output_tile_coord[0]*Th
    tile_bottom = tile_top+Th-1
    tile_left = output_tile_coord[1]*Tw
    tile_right = tile_left+Tw-1

    slice_info = [tile_left, tile_right, tile_top, tile_bottom ]
    real_index = slice_info
    while depth_convs > 0:
        padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
        pi = Pad_info(padding_info, slice_info, internal_expand, real_index)
        info_dict[depth_convs] = pi
        depth_convs -= 1
        print("pd info ", padding_info)
        print("slice info ", slice_info)
        print("indexing {}:{}, {}:{}".format(slice_info[2],slice_info[3]+1, slice_info[0],slice_info[1]+1) )
    return info_dict









    #for single conv2d
# def get_input_tile(output_tile_coord: List, H: int, W: int, Th: int, Tw: int, ph: int, pw: int, input):
#     tile_top = output_tile_coord[0]*Th
#     tile_bottom = tile_top+Th-1
#     tile_left = output_tile_coord[1]*Tw
#     tile_right = tile_left+Tw-1

#     padding_info, slice_info = conv2d_padding_info([tile_top, tile_bottom, tile_left, tile_right], [Th, Tw], [H, W], [ph, pw])
#     input_tile = input[:, :, slice_info[2]:slice_info[3]+1, slice_info[0]:slice_info[1]+1]       #NCHW
    
#     # print(padding_info)
#     # print(slice_info)
#     pd = torch.nn.ConstantPad2d(padding_info, 0)
#     input_tile = pd(input_tile)

#     return input_tile



