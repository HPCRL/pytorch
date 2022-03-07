from torch.fx import symbolic_trace
from collections import OrderedDict
from .graph_node import ModuleNode, InputNode, OutputNode

def create_internal_graph(network, input_tensors, input_index, verbose=False):
    traced = symbolic_trace(network)

    # Convert the fx graph to an internal graph representation
    name_to_module = {}
    for name, module in network.named_modules():
        name_to_module[name] = module
    graph = []
    output_tensors = []
    node_to_output = OrderedDict()
    input_index = 0

    for fx_node in traced.graph.nodes:
        if verbose:
            print("fx_node++", fx_node, fx_node)
        if fx_node.op == "call_module":
            module_name = fx_node.target
            module = name_to_module[module_name]
            node = ModuleNode.construct_node(fx_node, module)
        elif fx_node.op == "placeholder":
            node = InputNode(fx_node)
        # elif fx_node.op == "get_attr":
        #     node = AttributeNode(fx_node, self.model)
        # elif fx_node.op == "call_function" or fx_node.op == "call_method":
        #     node = FunctionNode.construct_node(fx_node)
        elif fx_node.op == "output":
            node = OutputNode(fx_node)
        else:
            assert 0, f"Unknown operator type: {fx_node.op}"
        graph.append(node)

    #Tensor shape propogation 
    output_tensors = []
    node_to_output = OrderedDict()
    input_index = 0

    for node in graph:
        if verbose:
            print(f"{node.ir_string}")
        if isinstance(node, InputNode):
            node_output = node.genGraph(input_tensors, input_index)
            input_index += 1
        elif isinstance(node, OutputNode):
            node.genGraph(node_to_output, output_tensors)
            node_output = None
        else:
            node_output = node.genGraph(node_to_output)

        # Save the node output for later nodes
        if node_output is not None:
            node_to_output[node.name] = node_output

    print(node_to_output)
    return output_tensors