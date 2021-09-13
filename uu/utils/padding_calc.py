import torch
from typing import Dict, List
from torch.autograd.variable import Variable
from uu.layers import maxpool2d, conv2d
import pdb
import math

def conv1d_padding_info(tile_indx: List, prb_size: List, pads: List):
    return

def conv2d_padding_info(tile_indx: List, prb_size: List, pads: List):
    #pdb.set_trace()
    H = prb_size[0]
    W = prb_size[1]

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]

    #print("index", tile_left, tile_right, tile_top, tile_bottom, )
    pad_top = max(0-(tile_top-pads[0]), 0)
    pad_bottom = max((tile_bottom+pads[0])-(H-1), 0)
    pad_left = max(0-(tile_left-pads[1]), 0)
    pad_right = max((tile_right+pads[1])-(W-1), 0)
    # padding 
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]

    #print(padding_info)
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

    # left , right, top, bottom
    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]
    return padding_info, slice_info, internal_expand, real_index

# might need to create a memo structure. 
class Pad_info:
    def __init__(self, coord, ordering_info, cur_output_shape, padding_info, slice_info, internal_expand, real_index, opname):
        self.coord = coord
        # self.ordering_info = ordering_info # [seg_id, position(0 base), depth(0 base)]
        self.cur_output_shape = cur_output_shape # current op produce [problem, tile] size; use it to remodify fwd
        self.padding_info = padding_info
        self.slice_info = slice_info
        self.internal_expand = internal_expand
        self.real_index = real_index
        self.opname = opname

    def __repr__(self) -> str:
        rep = self.opname + ' PI( <' + "".join([str(x)+"," for x in self.coord]) + '>,\n <' + \
                    "".join([str(x)+"," for x in self.cur_output_shape]) + '>,\n <padding ' + \
                    "".join([str(x)+"," for x in self.padding_info]) + '>,\n <sliidx ' + \
                    "".join([str(x)+"," for x in self.slice_info]) + '>, \n <internal ' + \
                    "".join([str(x)+"," for x in self.internal_expand]) + '>, \n <ridx' + \
                    "".join([str(x)+"," for x in self.real_index]) + '>, \n)' + '\n' 
        return rep

def compute_info_beta(output_tile_coord: List, input_shape, output_shape, nTh, nTw, stream_structure, shape_dict) -> Dict:
    list_op__in_chckp_seg = []
    has_maxpool = False
    for op in stream_structure._modules.values():
        if isinstance(op, maxpool2d.cMaxPool2d):
            has_maxpool = True
        list_op__in_chckp_seg.append(op)
        print("hash", id(op))

    print("op_list_in_seg", list_op__in_chckp_seg)
    
    if has_maxpool:
        f_info = compute_fwd_info_beta(output_tile_coord, output_shape, nTh, nTw, list_op__in_chckp_seg, shape_dict)
        b_info = compute_bwd_info_beta(output_tile_coord, input_shape, nTh, nTw, list_op__in_chckp_seg, shape_dict)
    else:
        f_info = compute_fwd_info_beta(output_tile_coord, output_shape, nTh, nTw, list_op__in_chckp_seg, shape_dict)

    
    # # print(f_info)
    # # print(b_info)
    # info = {**f_info, **b_info}
    # return info




def compute_fwd_info_beta(output_tile_coord: List, output_shape, nTh, nTw, list_op__in_chckp_seg) -> Dict:
    # stream_structure is list of ops
    # compute fwd is from the last stage
    list_op__in_chckp_seg.reverse()
    fwd_info_dict = {}
    with torch.no_grad():
        # calculate disjoint output tile first
        # no partial tile here
        H = output_shape[0]
        W = output_shape[1]
        
        op_idex = 0
        for op in list_op__in_chckp_seg:
            print("current op", id(op))
            Th = H // nTh
            Tw = W // nTw
            cur_output_shape = [H, W, Th, Tw]
            if isinstance(op, conv2d.TiledConv2d):
                if op_idex == 0:    # the very first one, compute info manually; 
                    tile_top = output_tile_coord[0]*Th
                    tile_bottom = tile_top+Th-1
                    tile_left = output_tile_coord[1]*Tw
                    tile_right = tile_left+Tw-1
                    slice_info = [tile_left, tile_right, tile_top, tile_bottom]
                    real_index = slice_info


                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
                opname = "conv2d"+id(op)
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, slice_info, internal_expand, real_index, opname)
                fwd_info_dict[id(op)] = pi  # insert into info_dict
            elif isinstance(op, maxpool2d.cMaxPool2d):
                rule = lambda x: 0 if x < 0 else x
                real_index = list(map(rule, real_index))

                # get a logic global view
                s_real_index = real_index.copy()
                s_real_index[1] = min(W-1, s_real_index[1] +1)
                s_real_index[3] = min(H-1, s_real_index[3] +1)
                padding_info = None
                internal_expand = None
                slice_info = None
                opname = "maxpool2d"+id(op)
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, slice_info, internal_expand, s_real_index, opname)
                fwd_info_dict[id(op)] = pi # insert into info_dict
                #print("pp real_index ", s_real_index)
                mxp_stride = op.stride[0]
                real_index = [x*mxp_stride for x in real_index]
                H = H * mxp_stride
                W = W * mxp_stride
                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1] +1)
                real_index[3] = min(H-1, real_index[3] +1)
            else:
                None
   
    return fwd_info_dict


def compute_bwd_info_beta(output_tile_coord: List, output_shape, nTh, nTw, list_op__in_chckp_seg) -> Dict:
    bwd_info_dict = {}
    with torch.no_grad():
        # calculate disjoint output tile first
        # no partial tile here
        H = output_shape[0]
        W = output_shape[1]
        
        Th = H // nTh
        Tw = W // nTw
        tile_top = output_tile_coord[0]*Th
        tile_bottom = tile_top+Th-1
        tile_left = output_tile_coord[1]*Tw
        tile_right = tile_left+Tw-1
        slice_info = [tile_left, tile_right, tile_top, tile_bottom]
        real_index = slice_info

        for op in list_op__in_chckp_seg:
            print("current op", id(op))

            cur_output_shape = [H, W, Th, Tw]
            if isinstance(op, conv2d.TiledConv2d):
                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
                opname = "conv2d"+id(op)
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, slice_info, internal_expand, real_index, opname)
                bwd_info_dict[id(op)] = pi  # insert into info_dict
            elif isinstance(op, maxpool2d.cMaxPool2d):
                rule = lambda x: 0 if x < 0 else x
                real_index = list(map(rule, real_index))

                # get a logic global view
                s_real_index = real_index.copy()
                s_real_index[1] = min(W-1, s_real_index[1] +1)
                s_real_index[3] = min(H-1, s_real_index[3] +1)
                padding_info = None
                internal_expand = None
                slice_info = None
                opname = "maxpool2d"+id(op)
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, slice_info, internal_expand, s_real_index, opname)
                bwd_info_dict[id(op)] = pi # insert into info_dict
                #print("pp real_index ", s_real_index)
                mxp_stride = op.stride[0]
                real_index = [x*mxp_stride for x in real_index]
                H = math.ceil(H / mxp_stride)
                W = math.ceil(W / mxp_stride)



                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1] +1)
                real_index[3] = min(H-1, real_index[3] +1)
            else:
                None
   
    return bwd_info_dict

def get_input_tile(info:Dict, input, depth: int):
    input_tile = None
    #print("depth", depth)
    with torch.no_grad():
        pi = info[depth]
        #padding_info = pi.padding_info
        slice_info = pi.slice_info
        input_tile = input[:, :, slice_info[2]:slice_info[3]+1, slice_info[0]:slice_info[1]+1]       #NCHW
        # print(" pi", pi)
        # pd = torch.nn.ConstantPad2d(padding_info, 0)
        # input_tile = pd(input_tile)
    
    input_tile.requires_grad = input.requires_grad
    assert input_tile is not None

    return Variable(input_tile, requires_grad = True)


def recreate_input_tile(info:Dict, input, depth: int):
    # print("recreate_input_tile next depth", depth)
    # peek current conv if it is the first one after a maxp
    cur_depth = depth+1
    c_pi = info[cur_depth]
    if c_pi.ordering_info[1] == 0:
        # if it is the first conv, do nothing on grad_out
        input_tile = input
        #print("== inputtile for next", input_tile.size(), input_tile)
    else:
        # if not the first conv, produce new input_tile
        pi = info[depth]
        padding_info = pi.padding_info
        #shifting tile to extract
        input_shape = input.size()
        top = padding_info[2]
        bottom = input_shape[2]-padding_info[3]
        left = padding_info[0]
        right = input_shape[3]-padding_info[1]
        # print("\n===\n")
        # print(input_shape)
        # print(padding_info)
        # print(slice_info)
        # print("top, bottom, left, right " , top, bottom, left, right)
        # print("\n===\n")
        input_tile = input[:, :, top:bottom, left:right]       #NCHW
        # print("== inputtile for next", input_tile.size(), input_tile)
        # print(padding_info)
        pd = torch.nn.ConstantPad2d(padding_info, 0)
        input_tile = pd(input_tile)

    return input_tile


def recreate_input_tile_f(info:Dict, input, depth: int):
    pi = info[depth]
    padding_info = pi.padding_info
    #shifting tile to extract
    input_shape = input.size()
    top = padding_info[2]
    bottom = input_shape[2]-padding_info[3]
    left = padding_info[0]
    right = input_shape[3]-padding_info[1]
    # print("\n===\n")
    # print(input_shape)
    # print(padding_info)
    # print(slice_info)
    # print("top, bottom, left, right " , top, bottom, left, right)
    # print("\n===\n")
    
    input_tile = input[:, :, top:bottom, left:right]       #NCHW
    #print("== inputtile for next", input_tile)
    #print(padding_info)
    pd = torch.nn.ConstantPad2d(padding_info, 0)
    input_tile = pd(input_tile)

    return input_tile



