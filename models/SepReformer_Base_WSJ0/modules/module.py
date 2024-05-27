import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .network import *


class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x
    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, spk_split_stage: dict, simple_fusion:dict, dec_stage: dict):
        super().__init__()
        
        class RelativePositionalEncoding(torch.nn.Module):
            def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
                super().__init__()
                self.in_channels = in_channels
                self.num_heads = num_heads
                self.embedding_dim = self.in_channels // self.num_heads
                self.maxlen = maxlen
                self.pe_k = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim)
                self.pe_v = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim) if embed_v else None
            
            def forward(self, pos_seq: torch.Tensor):
                pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
                pos_seq += self.maxlen
                pe_k_output = self.pe_k(pos_seq)
                pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
                return pe_k_output, pe_v_output
        
        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()
                                
                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        """Construct an EncoderLayer object."""
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()
                    
                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                
                self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None
                
            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                # [BK, S, N]
                return x, skip
        
        class SpkSplitStage(torch.nn.Module):
            def __init__(self, in_channels: int, num_spks: int):
                super().__init__()
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4*in_channels*num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2*in_channels*num_spks, in_channels*num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.num_spks = num_spks
                
            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                B, _, T = x.shape
                x = x.view(B*self.num_spks,-1, T).contiguous()
                x = self.norm(x)
                return x
        
        class SepDecStage(torch.nn.Module):
            def __init__(self, num_spks: int, global_blocks: dict, local_blocks: dict, spk_attention: dict):
                super().__init__()
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.spk_attn_1 = SpkAttention(**spk_attention)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.spk_attn_2 = SpkAttention(**spk_attention)
                
                self.g_block_3 = GlobalBlock(**global_blocks)
                self.l_block_3 = LocalBlock(**local_blocks)
                self.spk_attn_3 = SpkAttention(**spk_attention)
                
                self.num_spk = num_spks
            
            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                # [BS, K, H]
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_1(x, self.num_spk)
                
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_2(x, self.num_spk)
                
                x = self.g_block_3(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_3(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_3(x, self.num_spk)
                
                skip = x
                
                return x, skip
        
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(SepEncStage(**enc_stage, down_conv=True))
        
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        
        # Temporal Expanding Part
        self.simple_fusion = torch.nn.ModuleList([])
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.simple_fusion.append(torch.nn.Conv1d(in_channels=simple_fusion['out_channels']*2,out_channels=simple_fusion['out_channels'], kernel_size=1))
            self.dec_stages.append(SepDecStage(**dec_stage))
    
    def forward(self, input: torch.Tensor):
        '''input: [B, N, L]'''
        # feature projection
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        # Temporal Contracting Part
        pos_seq = torch.arange(0, len_x//2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip_ = self.spk_split_block(skip_)
            skip.append(skip_)
        x, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x) # B, 2F, T
        
        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = torch.nn.functional.upsample(x, skip[idx_en].shape[-1])
            x = torch.cat([x,skip[idx_en]],dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)
        
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        #  (B, T) or (B, 1, T)
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int, masking: bool = False):
        super().__init__()
        # feature expansion back
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask="ReLU", concat_opt=None)
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4*out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[...,:input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        
        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0,1).contiguous()
            input = input.view(B*self.num_spks, N, L)
            x = self.spe_block(x, input)
        
        x = x.view(B, self.num_spks, N, L)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # x: [B, N, L]
        if x.dim() not in [2, 3]: raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x