import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np
from mamba_ssm import Mamba
from typing import Callable, Optional
from layers.Short_encoder import Short_encoder
from layers.LWT_layers import series_decomp
from layers.Long_encoder import Long_encoder
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    SST (State Space Transformer)
    """
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        super().__init__()
        
        # mamba load parameters
        long_context_window = configs.seq_len
        m_layers = configs.m_layers
        d_state = configs.d_state
        d_conv = configs.d_conv
        m_patch_len = configs.m_patch_len
        m_stride = configs.m_stride

        # lwt load parameters
        c_in = configs.enc_in
        context_window = configs.label_len
        self.label_len = configs.label_len
        target_window = configs.pred_len
        local_ws = configs.local_ws
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        concat = configs.concat

        # Patching
        m_patch_num = int((long_context_window - m_patch_len)/m_stride + 1) + 1 if padding_patch == 'end' \
                                else int((long_context_window - m_patch_len)/m_stride + 1)
        patch_num = int((context_window - patch_len)/stride + 1) + 1 if padding_patch == 'end' \
                                else int((context_window - patch_len)/stride + 1)

        # head load parameters
        c_out = configs.c_out
        m_head_nf = d_model * m_patch_num
        t_head_nf = d_model * patch_num
        head_nf = d_model * (m_patch_num + patch_num)

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # mamba
        self.long_encoder = Long_encoder(
                                  c_in=c_in, long_context_window = long_context_window, 
                                  target_window=target_window, m_patch_len=m_patch_len, m_stride=m_stride, 
                                  m_layers=m_layers, d_model=d_model,
                                  d_ff=d_ff, norm=norm,
                                  dropout=dropout, act=act, 
                                  pre_norm=pre_norm,
                                  d_state=d_state, d_conv=d_conv,
                                  fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)


        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = Short_encoder(
                                  c_in=c_in, context_window = context_window, target_window=target_window, 
                                  local_ws=local_ws, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = Short_encoder(
                                  c_in=c_in, context_window = context_window, target_window=target_window,
                                  local_ws=local_ws, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = Short_encoder(
                                    c_in=c_in, context_window = context_window, target_window=target_window, 
                                  local_ws=local_ws, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        # Router
        self.router = Router(long_context_window=long_context_window, context_window=context_window, c_in=c_in, d_model=d_model)
        self.head = Fusion_Head(concat=concat, individual=individual, \
                              c_in=c_in, c_out=c_out, nf=head_nf, m_nf=m_head_nf, t_nf=t_head_nf,
                              target_window=target_window, head_dropout=head_dropout)

    def forward(self, long, long_mark, short_mark, self_mask=None):
        # norm
        if self.revin: 
            long = self.revin_layer(long, 'norm')
        
        m_weight, t_wieght = self.router(long)
        short = long[:, -self.label_len:, :]

        # mamba
        long = long.permute(0,2,1)    # x: [Batch, Channel, Input length]
        long = self.long_encoder(long) # x: [Batch, Channel, d_model, m_patch_num]
        # lwt
        if self.decomposition:
            res_init, trend_init = self.decomp_module(short)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            short = res + trend
            short = short.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            short = short.permute(0,2,1)    # x: [Batch, Channel, Input length]
            short = self.model(short)       # x: [Batch, Channel, d_model, patch_num]
        
        # fusion
        long_short = self.head(long, short, m_weight, t_wieght)
        long_short = long_short.permute(0,2,1)

        # denorm
        if self.revin: 
            long_short = self.revin_layer(long_short, 'denorm')

        return long_short

class Router(nn.Module):
    """
    Long-short router
    """
    def __init__(self, long_context_window, context_window, c_in, d_model, bias=True):
        super().__init__()

        # router
        self.context_window = context_window

        # project
        self.W_P = nn.Linear(c_in, d_model, bias=bias)
        self.flatten = nn.Flatten(start_dim=-2)
        # weighter
        self.W_w = nn.Linear(long_context_window*d_model, 2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long): # x: [Batch, Input length, Channel]
        # project
        x = self.W_P(long)
        x = self.flatten(x)

        # weighter
        prob = self.softmax(self.W_w(x))
        m_weight, t_weight = prob[:,0], prob[:,1]

        return m_weight, t_weight

class Fusion_Head(nn.Module):
    """
    Long-short range fusion head
    """
    def __init__(self, concat, individual, c_in, c_out, nf, m_nf, t_nf, target_window, head_dropout=0):
        super().__init__()

        #head
        self.concat = concat
        self.individual = individual
        self.c_in = c_in
        self.c_out = c_out
        self.target_window = target_window
        if self.concat:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                for i in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
        else:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                self.long_to_shorts = nn.ModuleList()
                for i in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.long_to_shorts.append(nn.Linear(m_nf, t_nf))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.long_to_short = nn.Linear(m_nf, t_nf) 
                self.linear = nn.Linear(t_nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, long, short, m_weight, t_weight):
        if self.concat:
            if self.individual:
                long_short_out = []
                for i in range(self.c_in):
                    long = self.flattens[i](long[:,i,:,:]) # z: [bs x d_model * patch_num]
                    short = self.flattens[i](short[:,i,:,:])
                    long_short = torch.cat((m_weight*long, t_weight*short), 1)
                    long_short = self.linears[i](long_short) # z: [bs x target_window]
                    long_short = self.dropouts[i](long_short) 
                    long_short_out.append(long_short)
                long_short = torch.stack(long_short_out, dim=1)                 # x: [bs x nvars x target_window]
            else:
                long, short = self.flatten(long), self.flatten(short) # x: [bs x nvars x d_model * patch_num]
                long_short = torch.cat((torch.mul(m_weight.view(-1,1,1), long), 
                                        torch.mul(t_weight.view(-1,1,1), short)), 2)
                long_short = self.linear(long_short) # x: [bs x nvars x target_window]
                long_short = self.dropout(long_short)
        else:
            if self.individual:
                long_short_out = []
                for i in range(self.c_in):
                    long = self.flattens[i](long[:,i,:,:]) # z: [bs x d_model * patch_num]
                    short = self.flattens[i](short[:,i,:,:])
                    long_short = m_weight*self.long_to_shorts[i](long) + t_weight*short 
                    long_short = self.linears[i](long_short) # z: [bs x target_window]
                    long_short = self.dropouts[i](long_short) 
                    long_short_out.append(long_short)
                long_short = torch.stack(long_short_out, dim=1)                 # x: [bs x nvars x target_window]
            else:
                long, short = self.flatten(long), self.flatten(short) # x: [bs x nvars x d_model * patch_num]
                long_short = m_weight*self.long_to_short(long) + t_weight*short 
                long_short = self.linear(long_short) # x: [bs x nvars x target_window]
                long_short = self.dropout(long_short)

        return long_short
