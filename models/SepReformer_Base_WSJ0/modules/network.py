import torch
import math
import numpy
from utils.decorators import *


class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,1,input_size)*Layer_scale_init, requires_grad=True)
        else:
            raise("Could you check your network, please? , You idiot??")
    
    def forward(self, x):
        return x*self.layer_scale

class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask='Sigmoid', **options):
        super(Masking, self).__init__()
        
        self.options = options
        if self.options['concat_opt']:
            self.pw_conv = torch.nn.Conv1d(input_dim*2, input_dim, 1, stride=1, padding=0)

        if Activation_mask == 'Sigmoid':
            self.gate_act = torch.nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = torch.nn.ReLU()
            

    def forward(self, x, skip):
   
        if self.options['concat_opt']:
            y = torch.cat([x, skip], dim=-2)
            y = self.pw_conv(y)
        else:
            y = x
        y = self.gate_act(y) * skip

        return y


class FFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels*6))
        self.depthwise = torch.nn.Conv1d(in_channels*6, in_channels*6, 3, padding=1, groups=in_channels*6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels*3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
        
    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + y*self.Layer_scale


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention layer.
        :param int n_head: the number of head s
        :param int n_feat: the number of features
        :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head: int, in_channels: int, dropout_rate: float, Layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head # We assume d_v always equals d_k
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x, pos_k, mask):
        """
        Compute 'Scaled Dot Product Attention'.
            :param torch.Tensor mask: (batch, time1, time2)
            :param torch.nn.Dropout dropout:
            :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
            weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))*self.Layer_scale  # (batch, time1, d_model)


class ConvLocalSelfAttention(torch.nn.Module):
    def __init__(self, in_channels, num_heads, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels*2)
        self.GLU = torch.nn.GLU()
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, 65, padding='same', groups=self.in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 2*in_channels)        
        self.BN = torch.nn.BatchNorm1d(2*in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(2*in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.GLU(y)
        y = y.permute([0, 2, 1]) # B, F, T
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1) # B, T, 2F
        y = self.linear2(y)
        y = y.permute(0, 2, 1) # B, T, 2F
        y = self.BN(y)
        y = y.permute(0, 2, 1) # B, T, 2F        
        y = self.linear3(y)
        
        return x + y*self.Layer_scale