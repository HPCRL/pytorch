import os
def make_graph(gr):
    import graphviz
    dot = graphviz.Digraph(format='svg', graph_attr={'labelloc': 't'})

    nodes = {}
    for i in gr.inputs():
        nname = i.debugName()
        label = nname.split('.')[0]
        nodes[nname] = (nname, dot)
        dot.node(nname, label, color='blue')

    unseen_ops = {'prim::ListConstruct', 'aten::index', 
                  'aten::size', 'aten::slice', 'aten::unsqueeze', 'aten::squeeze',
                  'aten::to', 'aten::view', 'aten::permute', 'aten::transpose', 'aten::contiguous',
                  'aten::permute', 'aten::Int', 'prim::TupleUnpack', 'prim::ListUnpack', 'aten::unbind',
                  'aten::select', 'aten::detach', 'aten::stack', 'aten::reshape', 'aten::split_with_sizes',
                  'aten::cat', 'aten::expand', 'aten::expand_as', 'aten::_shape_as_tensor',
                  'aten::_size_if_not_equal', 'prim::BroadcastSizes',
                  'prim::Constant',
                  }

    def process_block(nodeit, dot):
        firstnode = None
        lastnode = None
        for n in nodeit:
            k = n.kind()
            outs = list(n.outputs())
            inps = list(n.inputs())
            type_outs = [o.type().kind() for o in outs]
            type_inps = [o.type().kind() for o in inps]
            if k == 'prim::If':
                label = 'If'
                nname = outs[0].debugName()
                for i in inps:
                    src, srcdot = nodes.get(i.debugName(), (None, None))
                    if src is not None:
                        srcdot.edge(src, nname + '_in')
                dot.node(nname + '_in', 'If', shape='diamond')
                dot.node(nname, '', width='0.1', height='0.1')
                dot.edge(nname + '_in', nname, style='invis')
                nodes[nname] = (nname, dot)
                bl = list(n.blocks())
                for i, b in enumerate(bl):
                    with dot.subgraph(name=f"cluster_{nname}_{i}", graph_attr={'label':''}) as sub_dot:
                        firstnode, lastnode = process_block(b.nodes(), sub_dot)
                    dot.edge(nname + '_in', firstnode, label="yn"[i])
                    dot.edge(lastnode, nname)
                if firstnode is None:
                    firstnode = nname + '_in'
                lastnode = nname
            elif k == 'prim::DifferentiableGraph':
                label = 'DifferentiableGraph'
                nname = outs[0].debugName()
                nodes[nname] = (nname, dot)
                sg = n.g('Subgraph')
                nis = list(n.inputs())
                sgis = list(sg.inputs())
                assert len(nis) == len(sgis)
                for ni, sgi in zip(nis, sgis):
                    if ni.debugName() in nodes:
                        nodes[sgi.debugName()] = nodes[ni.debugName()]
                with dot.subgraph(name=f"cluster_{nname}", graph_attr={
                    'label': 'DifferentiableGraph', 'labelloc':'b', 'labeljust':'r'}) as sub_dot:
                    firstnode, lastnode = process_block(sg.nodes(), sub_dot)
                nos = list(n.outputs())
                sgos = list(sg.outputs())
                assert len(nos) <= len(sgos)
                for no, sgo in zip(nos, sgos):
                    if sgo.debugName() in nodes:
                        nodes[no.debugName()] = (nodes[sgo.debugName()][0], dot)
            elif k not in unseen_ops:
                if k == 'prim::CallMethod':
                    label = 'call ' + next(n.inputs()).node().s("name")
                else:
                    label = k.replace('aten::', '').replace('prim::', '')
                    label = label + " :- " + "\n"+str(n.getModuleHierarchy())
                nname = outs[0].debugName()
                print("nname", nname, label, n.getModuleHierarchy())

                dot.node(nname, label, shape='box', style='rounded')
                for o in outs:
                    nodes[o.debugName()] = (nname, dot)
                for i in inps:
                    src, srcdot = nodes.get(i.debugName(), (None, None))
                    if src is not None:
                        srcdot.edge(src, nname)
                if firstnode is None:
                    firstnode = nname
                lastnode = nname
        return firstnode, lastnode

    process_block(gr.nodes(), dot)
    dot.node('.outputs', 'outputs', color='blue')
    for i, o in enumerate(gr.outputs()):
        src, srcdot = nodes.get(o.debugName(), (None, None))
        if src is not None:
            dot.edge(src, '.outputs')

    return dot


def save_graph(dot, path, format="pdf"):     

    dot.format = format
    directory, file_name = os.path.split(path)
    file_name = file_name.replace("." + format, "")
    dot.render(file_name, directory=directory, cleanup=True)