import torch
from typing import Dict, List
from torch.autograd.variable import Variable
from uu.layers import maxpool2d, conv2d
import pdb
import math

def conv1d_padding_info(tile_indx: List, prb_size: List, pads: List):
    return

# Assume conv2d input output are same shape
def conv2d_padding_info(tile_indx: List, none_tiled_output_shape, pads: List, stride, RS):
    #pdb.set_trace()
    oH = none_tiled_output_shape[2]
    oW = none_tiled_output_shape[3]
    iH = (oH-1)*stride+RS-2*pads[0]
    iW = iH     #only consider square
    # output view

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]
    #print("index", tile_left, tile_right, tile_top, tile_bottom)

    #here we only consider stride = 1
    pad_top = max(0-(tile_top-pads[0]), 0)
    pad_bottom = max((tile_bottom+pads[0])-(iH-1), 0)
    pad_left = max(0-(tile_left-pads[1]), 0)
    pad_right = max((tile_right+pads[1])-(iW-1), 0)
    # padding 0 element
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]
    #print(padding_info)

    # TODO: is it general??
    iexp_top = pads[0] if (tile_top-pads[0])>=0 else 0
    iexp_bottom = pads[0] if (tile_bottom+pads[0])<=(iH-1) else 0
    iexp_left = pads[1] if (tile_left-pads[1]) >= 0 else 0
    iexp_right = pads[1] if (tile_right+pads[1])<=(iW-1) else 0
    internal_expand = [iexp_left, iexp_right, iexp_top, iexp_bottom]

    input_top = max(0, (tile_top-pads[0]))
    input_bottom = min(iH-1, (tile_bottom+pads[0]))
    input_left = max(0, (tile_left-pads[1]))
    input_right = min(iW-1, (tile_right+pads[1]))
    #input_tile view, the 4 point(l,r,t,b) in input tenser. Value is include [l, r], [t, b]
    input_slice = [input_left, input_right, input_top, input_bottom]

    # left , right, top, bottom; the naming is misleading; it means the relative index of current input view in its parent's view.
    # real index can have negative value and larger than iH,iW value, since it shows info one level up. 
    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]
    # print("--tile_indx", tile_indx)
    # print("--input_slice", input_slice)
    # print("--real_index", real_index)
    return padding_info, input_slice, internal_expand, real_index

# might need to create a memo structure. 
class Pad_info:
    def __init__(self, coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, op_idex, local_idex):
        self.coord = coord
        # self.ordering_info = ordering_info # [seg_id, position(0 base), depth(0 base)]
        self.cur_output_shape = cur_output_shape # current op produce [problem, tile] size; use it to remodify fwd
        self.padding_info = padding_info
        self.input_slice = input_slice
        self.internal_expand = internal_expand
        self.real_index = real_index
        self.opname = opname
        self.op_idex = op_idex
        self.local_idex = local_idex

    def __repr__(self) -> str:
        rep = self.opname +"[" +str(self.op_idex)+","+str(self.local_idex) + "]" +' PI( <' + "".join([str(x)+"," for x in self.coord]) + '>,\n <otileshape ' + \
                    "".join([str(x)+"," for x in self.cur_output_shape]) + '>,\n <padding ' + \
                    "".join([str(x)+"," for x in self.padding_info]) + '>,\n <inpslidx ' + \
                    "".join([str(x)+"," for x in self.input_slice]) + '>, \n <internal ' + \
                    "".join([str(x)+"," for x in self.internal_expand]) + '>, \n <realidx ' + \
                    "".join([str(x)+"," for x in self.real_index]) + '>, \n)' + '\n' 
        return rep

def recompute_fwd_info(list_op__in_chckp_seg, op_indx):
    before_issue_maxpool = list_op__in_chckp_seg[0:op_indx]
    after_issue_maxpool = list_op__in_chckp_seg[op_indx+1]
    fwd_info_dict = {}
    return fwd_info_dict


def shape_compatible(fwd_out_shape, bwd_out_shape):
    if fwd_out_shape[0] >= bwd_out_shape[0] and fwd_out_shape[1] >= bwd_out_shape[1]:
        return True
    else:
        return False

def peek_position(stream_structure, op_idex):
    num_conv_in_seg = 0
    while op_idex < len(stream_structure):
        op = stream_structure[op_idex]
        if isinstance(op, conv2d.TiledConv2d):
            num_conv_in_seg += 1
        else:
            break
        op_idex += 1
    return num_conv_in_seg

def compute_info_beta(output_tile_coord: List, input_shape, output_shape, nTh, nTw, stream_structure, shape_dict) -> Dict:
    list_op__in_chckp_seg = []
    has_maxpool = False
    for op in stream_structure._modules.values():
        if isinstance(op, maxpool2d.cMaxPool2d):
            has_maxpool = True
        list_op__in_chckp_seg.append(op)
        #print("hash", id(op))

    #print("op_list_in_seg", list_op__in_chckp_seg)
    if has_maxpool:
        f_info = compute_fwd_info_beta(output_tile_coord, output_shape, nTh, nTw, list_op__in_chckp_seg.copy(), stream_structure, shape_dict)
        #print("f_info", f_info)
        print("------------------------------")
        b_info = compute_bwd_info_beta(output_tile_coord, input_shape, nTh, nTw, list_op__in_chckp_seg.copy(), shape_dict)
        #print("b_info", b_info)
        
        
        op_indx = len(list_op__in_chckp_seg)-1
        # reverse check if the op is the last maxpool
        while op_indx >= 0:
            op = list_op__in_chckp_seg[op_indx]
            if isinstance(op, maxpool2d.cMaxPool2d):
                fwd_out_shape = f_info[id(op)].cur_output_shape
                bwd_out_shape = b_info[id(op)].cur_output_shape
                if not shape_compatible(fwd_out_shape, bwd_out_shape):
                    break
            op_indx -= 1
        #check compatiblity, fwd tile >= bwd tile; heuristic if the last maxpool shape is fine, all previous should be fine.
        issue_maxpool = None
        if op_indx >= 0:
            issue_maxpool = list_op__in_chckp_seg[op_indx]
            print("issue", op_indx, issue_maxpool)
            f_info = recompute_fwd_info(list_op__in_chckp_seg, op_indx)
    else:
        f_info = compute_fwd_info_beta(output_tile_coord, output_shape, nTh, nTw, list_op__in_chckp_seg)

    
    # # print(f_info)
    # # print(b_info)
    info = [f_info, b_info]
    print(info)
    return info

def compute_fwd_info_beta(output_tile_coord: List, output_shape, nTh, nTw, list_op__in_chckp_seg, stream_structure,shape_dict) -> Dict:
    # stream_structure is list of ops
    # compute fwd is from the last stage
    list_op__in_chckp_seg.reverse()
    fwd_info_dict = {}
    with torch.no_grad():
        # calculate disjoint output tile first
        # no partial tile here
        H = output_shape[2]
        W = output_shape[3]
        op_idex = 0
        local_idex = 0  # if after maxpool, local_idex reset to 0
        peek_conv2d_pos = 0
        for op in list_op__in_chckp_seg:
            if isinstance(op, conv2d.TiledConv2d):
                if op_idex == 0:    # the very first one, compute info manually; 
                    Th = H // nTh
                    Tw = W // nTw
                    tile_top = output_tile_coord[0]*Th
                    tile_bottom = tile_top+Th-1
                    tile_left = output_tile_coord[1]*Tw
                    tile_right = tile_left+Tw-1
                    input_slice = [tile_left, tile_right, tile_top, tile_bottom]
                    real_index = input_slice
                
                if peek_conv2d_pos == 0:
                    peek_conv2d_pos = peek_position(stream_structure, op_idex)

                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                none_tiled_output_shape = shape_dict[id(op)].output_shape
                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                padding_info, input_slice, internal_expand, real_index = conv2d_padding_info(real_index, none_tiled_output_shape, [ph, pw], op.stride[0], op.kernel_size[0])
                opname = "conv2d"+str(id(op))
                local_idex = peek_conv2d_pos
                peek_conv2d_pos -= 1
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, op_idex, local_idex)
                fwd_info_dict[id(op)] = pi  # insert into info_dict
            
            elif isinstance(op, maxpool2d.cMaxPool2d):
                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                opname = "maxpool2d"+str(id(op))
                #produce input shape
                mxp_stride = op.stride[0]
                real_index = [x*mxp_stride for x in input_slice] # expand it and get parent index
                H = H * mxp_stride
                W = W * mxp_stride
                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1] +1) # +1 since 0-based 
                real_index[3] = min(H-1, real_index[3] +1)
                input_slice = real_index    # maxpooling no padding here.
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, op_idex, -1)
                fwd_info_dict[id(op)] = pi # insert into info_dict
            else:
                None
            op_idex += 1
            
    return fwd_info_dict


def compute_bwd_info_beta(output_tile_coord: List, input_shape, nTh, nTw, list_op__in_chckp_seg, shape_dict) -> Dict:
    bwd_info_dict = {}
    with torch.no_grad():
        H = input_shape[2]
        W = input_shape[3]
        
        op_idex = 0
        for op in list_op__in_chckp_seg:
            #print("current op", id(op))
            if isinstance(op, conv2d.TiledConv2d):
                if op_idex == 0:    # the very first one, compute info manually; 
                    Th = H // nTh
                    Tw = W // nTw
                    tile_top = output_tile_coord[0]*Th
                    tile_bottom = tile_top+Th-1
                    tile_left = output_tile_coord[1]*Tw
                    tile_right = tile_left+Tw-1
                    input_slice = [tile_left, tile_right, tile_top, tile_bottom]
                    real_index = input_slice

                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                none_tiled_input_shape = shape_dict[id(op)].input_shape
                padding_info, input_slice, internal_expand, real_index = conv2d_padding_info(real_index, none_tiled_input_shape, [ph, pw], op.stride[0], op.kernel_size[0])
                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                opname = "bk-conv2d"+str(id(op))
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, -100, -100)
                bwd_info_dict[id(op)] = pi  # insert into info_dict
            elif isinstance(op, maxpool2d.cMaxPool2d):
                # get a logic global view
                opname = "bk-maxpool2d"+str(id(op))
                maxp_stride = op.stride[0]
                #print("pp real_index ", s_real_index)
                real_index = [math.floor(x / maxp_stride) for x in input_slice]
                H = math.floor(H / maxp_stride)
                W = math.floor(W / maxp_stride)
                cur_output_shape = (real_index[1]-real_index[0]+1, real_index[3]-real_index[2]+1) # r-l, b-t
                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1])
                real_index[3] = min(H-1, real_index[3])
                input_slice = real_index    # maxpooling no padding here.
                # id need to rethink
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, -100, -100)
                bwd_info_dict[id(op)] = pi # insert into info_dict
            else:
                None
            op_idex += 1
   
    return bwd_info_dict


def get_input_tile(info:Dict, input, first_op_in_seg):
    input_tile = None
    #print("depth", depth)
    with torch.no_grad():
        pi = info[first_op_in_seg]
        #padding_info = pi.padding_info
        slice_info = pi.input_slice
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
    top = 0 + padding_info[2]
    bottom = input_shape[2]-padding_info[3]
    left = 0 + padding_info[0]
    right = input_shape[3]-padding_info[1]
    # print("\n===\n")
    # print(input_shape)
    # print(padding_info)
    # print(slice_info)
    # print("top, bottom, left, right " , top, bottom, left, right)
    # print("\n===\n")
    
    input_tile = input[:, :, top:bottom, left:right]       #NCHW, included index
    #print("== inputtile for next", input_tile)
    #print(padding_info)
    pd = torch.nn.ConstantPad2d(padding_info, 0)
    input_tile = pd(input_tile)
    return input_tile



