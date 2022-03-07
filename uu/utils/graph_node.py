from .type import (ActiMode, AggrMode, DataType, OpType,
                           ParameterSyncType, PoolType, enum_to_int,
                           enum_to_str, int_to_enum, str_to_enum)
from enum import Enum
import torch
import math

DEBUG_MODE=True

IR_DELIMITER = "; "
INOUT_NODE_DELIMITER = ','


class Comparator(Enum):
    EQ = 0
    GEQ = 1

class Node():
    """This base class represents a node in the model computational graph (to
    be used internally for PyTorch to FlexFlow conversion)."""
    def __init__(self, node):
        self.name = node.name
        self.op_type = None
        self._ir_string = None

    def __repr__(self):
        return f"{type(self).__name__}: {self.name}"

    def assert_num_args(self, num_args, cmp):
        if cmp == Comparator.EQ:
            assert len(self.innodes) == num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects {num_args}" \
                "arguments"
        elif cmp == Comparator.GEQ:
            assert len(self.innodes) >= num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects at least " \
                f"{num_args} arguments"
    
    @property
    def ir_string(self):
        """Returns the string representation of the node."""
        if self._ir_string is None:
            self.parse()
        return self._ir_string
    
    def parse(self):
        """Parses the node to populate ``self._ir_string`` with a string
        representation."""
        raise NotImplementedError
    
    def parse_inoutnodes(self, nodes):
        """Parses the given input or output nodes, and returns a string
        representation."""
        if nodes is None:
            return ""
        assert type(nodes) is list or type(nodes) is tuple or \
            type(nodes) is dict
        return INOUT_NODE_DELIMITER.join([node.name for node in nodes]) + \
            INOUT_NODE_DELIMITER


class InputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = None
        self.outnodes = node.users
        self.op_type = OpType.INPUT
        
        if DEBUG_MODE:
            self.parse()
            print("\nInit InputNode", node)
            print("Inodes ", self.parse_inoutnodes(self.innodes))
            print("Outodes ", self.parse_inoutnodes(self.outnodes))
            

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s) 
    
    def genGraph(self, input_tensors, input_index):
        # Here we just return shape of input tensor
        return input_tensors[input_index]

class OutputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = None
        self.op_type = OpType.OUTPUT
        

        if DEBUG_MODE:
            self.parse()
            print("Inodes ", self.parse_inoutnodes(self.innodes))
            print("Outodes ", self.parse_inoutnodes(self.outnodes))

    def parse(self):
        # TODO: Assumes only one output
        self.assert_num_args(1, Comparator.EQ)
        s = [self.name]
        if type(self.innodes[0]) is tuple:
            innodes = self.innodes[0]
            s.append(self.parse_inoutnodes(innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        else:
            s.append(self.parse_inoutnodes(self.innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s)
    
    def genGraph(self, node_to_output, output_tensors):
        return


    

class ModuleNode(Node):
    def __init__(self, node, module):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.module = module

    @staticmethod
    def construct_node(node, module):
        if type(module) is torch.nn.modules.conv.Conv2d:
            return Conv2dNode(node, module)
        # elif type(module) is torch.nn.modules.linear.Linear:
        #     return LinearNode(node, module)
        elif type(module) is torch.nn.modules.pooling.MaxPool2d:
            return Pool2dNode(node, module, PoolType.POOL_MAX)
        elif type(module) is torch.nn.modules.pooling.AvgPool2d:
            return Pool2dNode(node, module, PoolType.POOL_AVG)
        # elif type(module) is torch.nn.modules.pooling.AdaptiveAvgPool2d:
        #     return AdaptivePool2dNode(node, module, PoolType.POOL_AVG)
        # elif type(module) is torch.nn.modules.batchnorm.BatchNorm2d:
        #     return BatchNorm2dNode(node, module)
        # elif type(module) is torch.nn.modules.dropout.Dropout:
        #     return DropoutMNode(node, module)
        # elif type(module) is torch.nn.modules.flatten.Flatten:
        #     return FlattenMNode(node, module)
        # elif type(module) is torch.nn.modules.activation.ReLU:
        #     return ReLUMNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Sigmoid:
        #     return SigmoidNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Tanh:
        #     return TanhMNode(node, module)
        # elif type(module) is torch.nn.modules.activation.ELU:
        #     return ELUNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Softmax:
        #     return SoftmaxMNode(node, module)
        # elif type(module) is torch.nn.modules.normalization.LayerNorm:
        #     return LayerNormNode(node, module)
        # elif type(module) is torch.nn.Identity:
        #     return IdentityNode(node, module)
        # elif type(module) is torch.nn.GELU:
        #     return GeluMNode(node, module)
        # elif isinstance(module, torch.nn.Embedding):
        #     return EmbeddingNode(node, module)
        else:
            assert 0, f"Unknown module: {module}"

class Conv2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.CONV2D
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)
        if DEBUG_MODE:
            self.parse()
            print("\nInit Conv2dNode", node, module)
            print("Inodes ", self.parse_inoutnodes(self.innodes))
            print("Outodes ", self.parse_inoutnodes(self.outnodes)) 

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.out_channels))
        s.append(str(self.module.kernel_size[0]))
        s.append(str(self.module.kernel_size[1]))
        s.append(str(self.module.stride[0]))
        s.append(str(self.module.stride[1]))
        s.append(str(self.module.padding[0]))
        s.append(str(self.module.padding[1]))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        s.append(str(self.module.groups))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._ir_string = IR_DELIMITER.join(s)

    def shapeInferRule(self, input_tensor):
        stride = self.module.stride
        pad = self.module.padding
        N = input_tensor[0]
        C = input_tensor[1]
        H = input_tensor[2] 
        W = input_tensor[3] 
        RS = self.module.kernel_size[0]
        K = self.module.out_channels
        H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0]+1)
        W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1]+1)
        output_shape = (N, K, H, W)
        return output_shape


    def genGraph(self, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        return output_tensor

class Pool2dNode(ModuleNode):
    def __init__(self, node, module, pool_type):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

        if DEBUG_MODE:
            self.parse()
            print("\nInit Pool2dNode", node, module, pool_type)
            print("Inodes ", self.parse_inoutnodes(self.innodes))
            print("Outodes ", self.parse_inoutnodes(self.outnodes))
            
    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        # FIXME MaxPool2d supports ceil_mode
        s.append(str(self.module.kernel_size))
        s.append(str(self.module.stride))
        s.append(str(self.module.padding))
        s.append(str(enum_to_int(PoolType, self.pool_type)))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        self._ir_string = IR_DELIMITER.join(s)

    def shapeInferRule(self, input_tensor):
        stride = self.module.stride
        pad = self.module.padding
        N = input_tensor[0]
        C = input_tensor[1]
        H = input_tensor[2] 
        W = input_tensor[3] 
        RS = self.module.kernel_size[0]
        H = math.floor((H+2*pad-(RS-1)-1)/stride[0]+1)
        W = math.floor((W+2*pad-(RS-1)-1)/stride[1]+1)
        output_shape = (N, C, H, W)
        return output_shape


    def genGraph(self, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        return output_tensor