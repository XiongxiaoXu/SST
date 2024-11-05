import torch
import torch.nn as nn
import torch.nn.functional as F

class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, d_model, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.mamba(x)
        x = self.norm2(x)

        return x

class MA_Layer(nn.Module):
    def __init__(self, mamba, self_attention, d_model, dropout):
        super(MA_Layer, self).__init__()
        self.mamba = mamba
        self.self_attention = self_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x = x + self.mamba(x)
        x = self.norm1(x)

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm2(x)

        return x

class Mamba_Layer(nn.Module):
    def __init__(self, mamba, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)

        return x

class Long_Layer(nn.Module):
    def __init__(self, mamba, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(Long_Layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba = mamba
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.mamba(x)

        x = self.lin2(self.activation(self.lin1(x)))

        return x



