

import torch.nn as nn
import torch.nn.functional as F
from .auxiliary_nets import Encoder, ResEncoder


class ContSupNode(nn.Module):
    def __init__(self, n_nodes: int=1, x_shortcut: bool=False, n_res_shortcut: int=0,
                 wide_list=(3, 16, 16, 32, 64), size_list=(32, 32, 32, 16, 8), aux_net_widen: int=1):
        super(ContSupNode, self).__init__()
        # wide_list/size_list  ===> [x, h_1, h_2, ..., h_{node_idx+1}, ..., h_{n_res_shortcut}] <=== {length: n_res_shortcut+1}

        self.n_nodes = n_nodes
        self.n_res_shortcut = n_res_shortcut
        self.x_shortcut = x_shortcut

        for node_idx in range(n_nodes):
            exec('self.encoder_' + str(node_idx) +
                 '= Encoder(wide_list[node_idx+1], size_list[node_idx+1], widen=aux_net_widen)')
            if self.n_res_shortcut > 0:
                for res_sc_idx in range(n_res_shortcut):
                    if node_idx-res_sc_idx >= 0:
                        exec('self.resencoder_' + str(node_idx) + '_' + str(res_sc_idx) +
                            '= ResEncoder(wide_list[node_idx-res_sc_idx], wide_list[node_idx+1], size_list[node_idx+1], widen=aux_net_widen)')


    def forward(self, node_idx, x, context):
        ret = x.detach()
        new_context = context.copy()
        if self.x_shortcut:
            x_e = eval('self.encoder_' + str(node_idx))(context[-1].detach())
            ret += x_e
        if self.n_res_shortcut > 0:
            for res_sc_idx in range(self.n_res_shortcut):
                if context[res_sc_idx] != None:
                    ret += eval('self.resencoder_' + str(node_idx) + '_' + str(res_sc_idx))(context[res_sc_idx].detach())
                    if res_sc_idx != (self.n_res_shortcut - 1):
                        new_context[res_sc_idx+1] = context[res_sc_idx].detach()
                else:
                    if res_sc_idx != (self.n_res_shortcut - 1):
                        new_context[res_sc_idx+1] = context[res_sc_idx]
            new_context[0] = ret.detach()
        return ret, new_context

    def init_context(self, img):
        #### init_context ===> [x, None, None, ..., None, x] <=== {length: n_res_shortcut + 1}
        context = []
        if self.n_res_shortcut > 0:
            if self.x_shortcut: # skip x when encoder works
                context.append(None)
            else:
                context.append(img)
            for res_sc_idx in range(self.n_res_shortcut-1):
                context.append(None)
        if self.x_shortcut:
            context.append(img)

        return context