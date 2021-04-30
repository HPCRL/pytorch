import torch
import os
import re
from typing import List

THEMES = {
    "basic": {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },

}


class MetaNode():
    def __init__(self, uid: str, name: str, op, output_shape=None, input_shape_list: List[List]=None, output_shape_list: List[List]=None, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid
        self.name = name 
        self.op = op
        self.repeat = 1
        if output_shape:
            assert isinstance(output_shape, (tuple, list)),\
            "output_shape must be a tuple or list but received {}".format(type(output_shape))
        self.output_shape = output_shape

        self.input_shape_list = input_shape_list
        self.output_shape_list = output_shape_list
        self.params = params if params else {}
    
    def get_id(self) -> str:
        return self.id

    @property
    def title(self) -> str:
        """Returns detail info of a node"""
        title = self.name or self.op
        if "kernel_shape" in self.params.keys():
            # Kernel
            kernel = self.params["kernel_shape"]
            title += "x".join(map(str, kernel))
        if "stride" in self.params.keys():
            stride = self.params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                title += "/s{}".format(str(stride))
        #         # Transposed
        #         if node.transposed:
        #             name = "Transposed" + name


        return title

    def __repr__(self):
        args = (self.op, self.name, self.id, self.title, self.repeat)
        f = "<Node: op: {}, name: {}, id: {}, title: {}, repeat: {}"
        if self.output_shape:
            args += (str(self.output_shape),)
            f += ", shape: {:}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:}"
        f += ">"
        return f.format(*args)


class Graph():
    def __init__(self, model=None, args=None, input_names=None,
                transforms="default", framework_transforms="default",
                meaningful_ids=False):
        self.nodes = {}
        self.edges = []
        self.theme = THEMES["basic"]

    def id(self, node) -> str:
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def get_node_by_id(self, id_val) -> MetaNode:
        """Returns a unique node by identifier. """
        #print (self.nodes)
        for idv, n in self.nodes.items():
            if n.get_id() == id_val:
                return n
        return None

    def add_node(self, node):
        id = self.id(node)
        # assert(id not in self.nodes)
        self.nodes[id] = node

    def add_edge_by_id(self, vid1, vid2, label=None):
        # If the edge is already present, don't add it again.
        edge = (vid1, vid2, label)
        if edge not in self.edges:
            self.edges.append(edge)

    def outgoing(self, node) -> List[MetaNode]:
        """Returns nodes connecting out of the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges outgoing from this group but not incoming to it
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node) -> List[MetaNode]:
        """Returns nodes connecting to the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges incoming to this group but not outgoing from it
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming
    
    def remove(self, nodes):
        """Remove a node"""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            del self.nodes[k]

    def replace(self, nodes, node):
        """Replace nodes with node. Edges incoming to nodes[0] are connected to
        the new node, and nodes outgoing from nodes[-1] become outgoing from
        the new node."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # Is the new node part of the replace nodes (i.e. want to collapse
        # a group of nodes into one of them)?
        collapse = self.id(node) in self.nodes
        # Add new node and edges
        if not collapse:
            self.add_node(node)
        for in_node in self.incoming(nodes):
            # TODO: check specifically for output_shape is not generic. Consider refactoring.
            self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
        # Remove the old nodes
        for n in nodes:
            if collapse and n == node:
                continue
            self.remove(n)

    def build_dot(self):
        """Generate a GraphViz Dot graph.
        Returns a GraphViz Digraph object.
        """
        print(self.nodes)
        from graphviz import Digraph
        dot = Digraph()
        dot.attr("graph", 
                 bgcolor=self.theme["background_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"],
                 margin=self.theme["margin"],
                 rankdir="LR",
                 pad=self.theme["padding"])
        dot.attr("node", shape="box", 
                 style="filled", margin="0,0",
                 fillcolor=self.theme["fill_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])
        dot.attr("edge", style="solid", 
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])

        for k, n in self.nodes.items():
            label = "<tr><td cellpadding='6'>{}</td></tr>".format(n.title)
            if n.repeat > 1:
                label += "<tr><td align='right' cellpadding='2'>x{}</td></tr>".format(n.repeat)
            label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
            dot.node(str(k), label)
        for a, b, label in self.edges:
            if isinstance(label, (list, tuple)):
                label = "x".join([str(l or "?") for l in label])
            dot.edge(str(a), str(b), label)
        return dot

    def save(self, path, format="pdf"):
            dot = self.build_dot()
            dot.format = format
            directory, file_name = os.path.split(path)
            file_name = file_name.replace("." + format, "")
            dot.render(file_name, directory=directory, cleanup=True)

def dump_id_graph(graph: Graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))

def pytorch_id(node) -> str:
    """Returns a unique ID for a node."""
    return node.scopeName() + "/outputs/" + "/".join(["{}".format(o.unique()) for o in node.outputs()])

def get_output_shape(torch_node) -> List[List[int]]:
    #print("-- m ", str(next(torch_node.outputs())) ) 
    num_of_outputs = len(list(torch_node.outputs()))
    #print("num_of_outputs ", num_of_outputs)
    shape_list = []
    for output in list(torch_node.outputs()):
        try:
            shape = output.type().sizes()
        except:
            shape = None
        
        if shape is not None:
            shape_list.append(shape)
    return shape_list

def get_input_shape(torch_node) -> List[List[int]]:
    #print("-- m ", str(next(torch_node.outputs())) ) 
    num_of_inputs = len(list(torch_node.inputs()))
    #print("num_of_inputs ", num_of_inputs)
    shape_list = []
    for input in list(torch_node.inputs()):
        try:
            shape = input.type().sizes()
        except:
            shape = None
        
        if shape is not None:
            shape_list.append(shape)
    return shape_list   

def get_shape(torch_node) -> List[int]:
    #print ("---m ", torch_node)
    try:
        shape = torch_node.output().type().sizes()
    except:
        shape = None
    return shape


def build_graph(model=None, args=None, debug: bool=False):
    # Initialize an empty graph
    g = Graph()
    trace_graph, out = torch.jit._get_trace_graph(model, args)
    trace_graph = torch.onnx._optimize_trace(trace_graph, torch.onnx.OperatorExportTypes.RAW)

    fake_root_node = MetaNode(uid="Root51118102", name="Root", op=None, 
                       output_shape=args.size(), input_shape_list=[], output_shape_list=[], params=None)
    g.add_node(fake_root_node)

    if debug:
        dump_id_graph(trace_graph)

    for torch_node in trace_graph.nodes():
        # Op
        op = torch_node.kind()
        # Parameters
        try:
            params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        except:
            params = None
        # Inputs/outputs
        inputs = [i.unique() for i in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        output_shape = get_output_shape(torch_node)
        input_shape = get_input_shape(torch_node)

        if str(op).strip() == "prim::Constant":
            print ("SKIP")    
        else:
            print("-- op ", op)
            print("-- params ", params)
            print("-- inputs shape", input_shape)
            print("-- outputs shape", output_shape)
            print("-- inputs ", inputs)
            print("-- outputs ", outputs)
            
            shape = get_shape(torch_node)
            # Add node
            meta_node = MetaNode(uid=pytorch_id(torch_node), name=None, op=op, 
                        output_shape=shape, input_shape_list=input_shape, output_shape_list=output_shape, params=params)
            g.add_node(meta_node)
            # Add edges
            for target_torch_node in trace_graph.nodes():
                target_inputs = [i.unique() for i in target_torch_node.inputs()]
                if set(outputs) & set(target_inputs):
                    g.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
                if 0 in set(target_inputs):
                    root = g.get_node_by_id("Root51118102")
                    #print (root)
                    g.add_edge_by_id("Root51118102", pytorch_id(target_torch_node), root.output_shape)
    return g


