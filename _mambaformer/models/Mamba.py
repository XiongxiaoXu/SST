import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import Mamba_Layer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from mamba_ssm import Mamba

class Model(nn.Module):
    """
    Mamba
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

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

        self.mamba_layers = nn.ModuleList(
            [
                Mamba_Layer(
                    Mamba(configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv),
                    configs.d_model
                ) 
                for i in range(configs.d_layers)
            ]
        )
        self.out_proj=nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_dec, x_mark_dec):
        x = self.dec_embedding(x_dec, x_mark_dec)
        for i in range(self.configs.d_layers):
            x = self.mamba_layers[i](x)
        out = self.out_proj(x)

        return out