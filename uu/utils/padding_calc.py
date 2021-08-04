from torch.nn.modules import conv
import torch
from typing import Dict, List
from torch.autograd.variable import Variable
from torch.nn.modules.utils import _pair
import pdb

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
    def __init__(self, coord, ordering_info, pt_size, padding_info, slice_info, internal_expand, real_index):
        self.coord = coord
        self.ordering_info = ordering_info # [seg_id, position(0 base), depth(0 base)]
        self.pt_size = pt_size # [problem, tile] size
        self.padding_info = padding_info
        self.slice_info = slice_info
        self.internal_expand = internal_expand
        self.real_index = real_index

    def __repr__(self) -> str:
        rep = 'PI( <' + "".join([str(x)+"," for x in self.coord]) + '>,\n <' + \
                    "".join([str(x)+"," for x in self.ordering_info]) + '>,\n <p-tsize ' + \
                    "".join([str(x)+"," for x in self.pt_size]) + '>,\n <padding ' + \
                    "".join([str(x)+"," for x in self.padding_info]) + '>,\n <sliidx ' + \
                    "".join([str(x)+"," for x in self.slice_info]) + '>, \n <internal ' + \
                    "".join([str(x)+"," for x in self.internal_expand]) + '>, \n <ridx' + \
                    "".join([str(x)+"," for x in self.real_index]) + '>, \n)' + '\n'
        return rep

def compute_info(output_tile_coord: List, H: int, W: int, Th: int, Tw: int, ph: int, pw: int, input, num_convs: int) -> Dict:
    with torch.no_grad():
        info_dict = {}
        tile_top = output_tile_coord[0]*Th
        tile_bottom = tile_top+Th-1
        tile_left = output_tile_coord[1]*Tw
        tile_right = tile_left+Tw-1

        depth_convs = 0
        slice_info = [tile_left, tile_right, tile_top, tile_bottom ]
        real_index = slice_info
        org_size = [H, W, Th, Tw]

        #TODO: logic need to change. how to feed conv2d + pooling number
        while depth_convs < num_convs:
            padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
            ordering_info = [0, depth_convs]
            pi = Pad_info(output_tile_coord, ordering_info, org_size, padding_info, slice_info, internal_expand, real_index)
            info_dict[depth_convs] = pi
            depth_convs += 1
            # print("pd info ", padding_info)
            # print("slice info ", slice_info)
            # print("indexing {}:{}, {}:{}".format(slice_info[2],slice_info[3]+1, slice_info[0],slice_info[1]+1) )
    return info_dict

def compute_info_beta(output_tile_coord: List, H: int, W: int, nTh: int, nTw: int, ph: int, pw: int, stream_structure: List[_pair], num_conv:int, num_maxp) -> Dict:
    f_info = compute_fwd_info_beta(output_tile_coord, H, W, nTh, nTw, ph, pw, stream_structure, num_conv)
    b_info = compute_bwd_info_beta(output_tile_coord, H, W, nTh, nTw, ph, pw, stream_structure, num_conv, num_maxp)

    # print(f_info)
    # print(b_info)
    info = {**f_info, **b_info}
    return info




def compute_fwd_info_beta(output_tile_coord: List, H: int, W: int, nTh: int, nTw: int, ph: int, pw: int, stream_structure: List[_pair], num_conv:int) -> Dict:
    # stream_structure provide inforation in a tilable segment which presents num of continious conv2d and num of pooling. 
    # Store pair in an ordered list
    # exp, [(conv2d, 3), (pooling, 1), (conv2d, 2),....]
    with torch.no_grad():
        info_dict = {}
        depth_convs = 0
        c_seg = len(stream_structure)-1 # reversed order
        conv_g_id = 1 # reversed order and starting from 1. "1" is the last conv2d
        maxp_g_id = 1 # reversed order and starting from 1. "1" is the last conv2d

        stack = []  
        for s in stream_structure:
            stack.append(s)
            if s[0] == "pooling":
                H = H // 2
                W = W // 2

        real_index = []     # it is a iteratable variable
        while c_seg >= 0:
            p = stack.pop()
            if p[0] == "conv2d":
                seg_num_convs = p[1]
                while depth_convs < seg_num_convs: # depth in segment; if == 0 then last conv in the segment
                    Th = H // nTh
                    Tw = W // nTw
                    if conv_g_id == 1:
                        tile_top = output_tile_coord[0]*Th
                        tile_bottom = tile_top+Th-1
                        tile_left = output_tile_coord[1]*Tw
                        tile_right = tile_left+Tw-1
                        slice_info = [tile_left, tile_right, tile_top, tile_bottom]
                        real_index = slice_info

                    pt_size = [H, W, Th, Tw]
                    # print("W , H ", H, W)
                    # print("TW , TH ", Th, Tw)
                    #print("real_index ", real_index)

                    padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
                    ordering_info = [c_seg, seg_num_convs-depth_convs-1, depth_convs]
                    pi = Pad_info(output_tile_coord, ordering_info, pt_size, padding_info, slice_info, internal_expand, real_index)
                    info_dict[conv_g_id] = pi
                    
                    #print("input real_index ", real_index)
                    #print("conv_g_id - pi", conv_g_id, pi)
                    depth_convs += 1
                    conv_g_id += 1
            elif p[0] == "pooling":
                # 1) reset depth_convs to 0; 
                # 2) change H/W to half?? TODO: it is not general
                depth_convs = 0
                print("pp real_index ", real_index)
                rule = lambda x: 0 if x < 0 else x
                real_index = list(map(rule, real_index))

                # get a logic global view
                s_real_index = real_index.copy()
                s_real_index[1] = min(W-1, s_real_index[1] +1)
                s_real_index[3] = min(H-1, s_real_index[3] +1)
                info_dict[maxp_g_id+0.5] = s_real_index
                print("pp real_index ", s_real_index)


                real_index = [x*2 for x in real_index]
                H = H * 2
                W = W * 2
                
                real_index[1] = min(W-1, real_index[1] +1)
                real_index[3] = min(H-1, real_index[3] +1)
                # right and bottom need plus 1
                print("pp real_index left right top bot ", real_index)
                
                maxp_g_id += 1
            c_seg -= 1

    assert conv_g_id == num_conv+1
    return info_dict


def compute_bwd_info_beta(output_tile_coord: List, H: int, W: int, nTh: int, nTw: int, ph: int, pw: int, stream_structure: List[_pair], num_conv:int, num_maxp:int) -> Dict:
    with torch.no_grad():
        b_info_dict = {}
        depth_convs = 0
        c_seg = 0 
        conv_g_id = num_conv
        maxp_g_id = num_maxp

        #since doing backward padding computation, we start from very begining
        real_index = []     # it is a iteratable variable
        while c_seg < len(stream_structure):
            p = stream_structure[c_seg]
            if p[0] == "conv2d":
                seg_num_convs = p[1]
                depth_convs = seg_num_convs -1
                while depth_convs >= 0: # depth in segment; if == 0 then last conv in the segment
                    Th = H // nTh
                    Tw = W // nTw
                    if conv_g_id == num_conv:
                        tile_top = output_tile_coord[0]*Th
                        tile_bottom = tile_top+Th-1
                        tile_left = output_tile_coord[1]*Tw
                        tile_right = tile_left+Tw-1
                        slice_info = [tile_left, tile_right, tile_top, tile_bottom]
                        real_index = slice_info

                    pt_size = [H, W, Th, Tw]
                    # print("W , H ", H, W)
                    # print("TW , TH ", Th, Tw)
                    #print("real_index ", real_index)

                    padding_info, slice_info, internal_expand, real_index = conv2d_padding_info(real_index, [H, W], [ph, pw])
                    ordering_info = [c_seg, seg_num_convs-depth_convs-1, depth_convs]
                    pi = Pad_info(output_tile_coord, ordering_info, pt_size, padding_info, slice_info, internal_expand, real_index)
                    b_info_dict[-1*conv_g_id] = pi
                    
                    #print("input real_index ", real_index)
                    #print("conv_g_id - pi", conv_g_id, pi)
                    depth_convs -= 1
                    conv_g_id -= 1
            elif p[0] == "pooling":
                # 1) reset depth_convs to 0; 
                # 2) change H/W to half?? TODO: it is not general
                depth_convs = seg_num_convs -1
                #print("pp real_index ", real_index)
                real_index = [x // 2 for x in real_index]
                rule = lambda x: 0 if x < 0 else x
                real_index = list(map(rule, real_index))
                
                H = H // 2
                W = W // 2
                
                real_index[1] = min(W-1, real_index[1] +1)
                real_index[3] = min(H-1, real_index[3] +1)
                # right and bottom need plus 1
                print("pp back real_index left right top bottom", real_index)
                b_info_dict[-1*maxp_g_id-0.5] = real_index
                maxp_g_id -= 1
            c_seg += 1

    return b_info_dict


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



