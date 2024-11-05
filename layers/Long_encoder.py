# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.LWT_layers import *
from layers.RevIN import RevIN
from mamba_ssm import Mamba

# Cell
class Long_encoder(nn.Module):
    def __init__(self, c_in:int, long_context_window:int, target_window:int, m_patch_len:int, m_stride:int,
                 m_layers:int=3, d_model=128,
                 d_ff:int=256, norm:str='BatchNorm', dropout:float=0., act:str="gelu",
                 pre_norm:bool=False,
                 d_state:int=16, d_conv:int=4,
                 fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        # Patching
        self.m_patch_len = m_patch_len
        self.m_stride = m_stride
        self.padding_patch = padding_patch
        m_patch_num = int((long_context_window - m_patch_len)/m_stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, m_stride)) 
            m_patch_num += 1

        # Backbone 
        self.backbone = Mamba_Encoder(c_in, m_layers, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len) 
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.m_patch_len, step=self.m_stride)               # z: [bs x nvars x m_patch_num x m_patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x m_patch_len x m_patch_num]
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x m_patch_num]

        return z


class Mamba_Encoder(nn.Module):
    def __init__(self, c_in, m_layers, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len):
        super().__init__()

        W_inp_len = m_patch_len
        self.W_P = nn.Linear(W_inp_len, d_model)
        self.m_layers = m_layers

        self.mamba_layers = nn.ModuleList([Mamba_Encoder_Layer(d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len) for i in range(m_layers)])
    
    def forward(self, x):
        bs, nvars, m_patch_len, m_patch_num = x.shape
        x = x.permute(0,1,3,2) 
        x = x.view(bs*nvars, m_patch_num, m_patch_len) 
        # Input encoding
        x = self.W_P(x) # x: [(bs*nvars) x m_patch_num x d_model]

        for i in range(self.m_layers):
            x = self.mamba_layers[i](x)

        x = x.view(bs, nvars, m_patch_num, -1) 
        x = x.permute(0,1,3,2) # x: [bs x nvars x d_model x m_patch_num]

        return x


class Mamba_Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv)
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if act == "relu" else F.gelu

    def forward(self, x):
        x = self.mamba(x)
        x = self.lin2(self.act(self.lin1(x)))

        return x
    
