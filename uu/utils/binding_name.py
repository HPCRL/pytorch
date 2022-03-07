from torch.fx import symbolic_trace

def binding_module_name(network):
    symbolic_trace(network)