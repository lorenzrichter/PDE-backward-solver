import xerus as xe
################################################################################
#    Functions                                                                 #
################################################################################

def check_tt(tn):
    if not isinstance(tn, xe.TensorNetwork):
        return False
    left_link_to  = lambda node, pos: not node.neighbors[0].external and node.neighbors[0].other == pos
    right_link_to = lambda node, pos: not node.neighbors[-1].external and node.neighbors[-1].other == pos
    tt_component  = lambda node, pos: len(node.neighbors) == 3 and left_link_to(node, pos-1) and node.neighbors[1].external and right_link_to(node, pos+1)
    end_component = lambda node: node.tensorObject.dimensions == [1] and node.tensorObject[0] == 1
    if not end_component(tn.nodes[0]) and right_link_to(tn.nodes[0], 1):
        return False
    for pos,node in enumerate(tn.nodes[1:-1], start=1):
        if not tt_component(node, pos):
            return False
    return end_component(tn.nodes[-1]) and left_link_to(tn.nodes[-1], pos)


def tn2tt(tn):
    assert check_tt(tn)
    tt_list = [ttn.tensorObject for ttn in tn.nodes[1:-1]]
    tt = xe.TTTensor([t.dimensions[1] for t in tt_list])
    for pos in range(len(tt_list)):
        tt.set_component(pos, tt_list[pos])
    def inner(t1, t2):
        i, = xe.indices(1)
        return float(t1(i&0) * t2(i&0))
    tn_norm = xe.frob_norm(tn)**2
    try:
        error = abs(tn_norm + xe.frob_norm(tt)**2 - 2*inner(tn, tt))/tn_norm
    except ZeroDivisionError:
        error = 0
    if not error <= 1e-12:
        raise argparse.ArgumentTypeError("could not convert tensor network to tensor train network (relative error: {:.2e})".format(error))
    return tt
