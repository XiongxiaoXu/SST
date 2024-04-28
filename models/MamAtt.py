import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import MA_Layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np
from mamba_ssm import Mamba

class Model(nn.Module):
    """
    Mamba-Attention Architcture
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.MA_layers = nn.ModuleList(
            [
                MA_Layer(
                    Mamba(configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv),
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    dropout=configs.dropout,
                ) 
                for i in range(configs.d_layers)
            ]
        )
        self.out_proj=nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_dec, x_mark_dec, dec_self_mask=None):
        x = self.dec_embedding(x_dec, x_mark_dec)
        for i in range(self.configs.d_layers):
            x = self.MA_layers[i](x, dec_self_mask)
        out = self.out_proj(x)

        return out