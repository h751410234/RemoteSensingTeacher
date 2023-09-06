import torch
from torch import nn
import torch.nn.functional as F
#from einops import rearrange



class FSAS(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(FSAS, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(dim / self.num_attention_heads)

        self.query = nn.Linear(dim, dim)    # 256, 256
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(256)


    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   #
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]


    def forward(self, q,k,v):

        mixed_query_layer = self.query(q)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(k)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(v)   # [bs, seqlen, hid_size]

        #多头
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 32]
        key_layer = self.transpose_for_scores(mixed_key_layer)       # [bs, 8, seqlen, 32]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 32]

        #转换到频域
        q_fft = torch.fft.rfft2(query_layer.float())
        k_fft = torch.fft.rfft2(key_layer.float())

        #相乘
        xqk_ft = q_fft * k_fft
        attention = torch.fft.irfft2(xqk_ft)
        attention = torch.softmax(attention , dim=-1)
        output = value_layer * attention
        output = output.permute(0, 2, 1, 3).flatten(2).contiguous()

        #加入add and norm
        output_final = self.dropout(output) + v
        output_final = self.norm(output_final)

        return output_final


class LFFFN(nn.Module):
    def __init__(self, dim = 256):
        super(LFFFN, self).__init__()
        self.dim = dim
        self.hidden_dim = self.dim * 4
        self.learned_filter = nn.Parameter(torch.ones(self.hidden_dim  // 2 + 1))
        #---
        self.linear1 = nn.Linear(self.dim, self.hidden_dim)
        self.activation = F.gelu
        self.dropout1 = nn.Dropout(0.1)
        #---
        self.linear2 = nn.Linear(self.hidden_dim, self.dim)
        self.dropout2 = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        #1.
        __x = self.dropout1(self.activation(self.linear1(x)))
        __x = torch.fft.rfft2(__x.float())
        __x = __x * self.learned_filter
        __x = torch.fft.irfft2(__x)
        __x = self.linear2(__x)
        x = x + self.dropout2(__x)
        x = self.norm(x)
        return x


