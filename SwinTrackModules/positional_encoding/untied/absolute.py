import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from SwinTrackModules.positional_encoding.learned import Learned2DPositionalEncoder

class Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, w, h, scale=None, with_q=True, with_k=True):  
        super(Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, w, h)  
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:  
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:  
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        pos = self.norm(self.pos())  
        seq_len = pos.shape[0]  
        if self.pos_q_linear is not None and self.pos_k_linear is not None:  
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale  
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)  
            return pos_q, pos_k
        elif self.pos_q_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            return pos_q
        elif self.pos_k_linear is not None:
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_k
        else:
            raise RuntimeError


class dualImag_Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, wz, hz, wx, hx, scale=None, with_q=True, with_k=True):  
        super(dualImag_Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, wz+wx, hz+hx)  
        self.wz = wz
        self.wx = wx
        self.hz = hz
        self.hx = hx
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:  
            self.z_pos_q_linear = nn.Linear(dim, dim)
            self.x_pos_q_linear = nn.Linear(dim, dim)
        if with_k:  
            self.z_pos_k_linear = nn.Linear(dim, dim)
            self.x_pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        pos = self.norm(self.pos())  
        pos = pos.view(self.wz+self.wx, self.hz+self.hx, self.dim)  
        z_top_left_pos = pos[:self.wz, :self.hz, :]  
        x_bottom_right_pos = pos[self.wz:, self.hz:, :]  

        z_top_left_pos = z_top_left_pos.reshape(self.wz*self.hz, self.dim)  
        x_bottom_right_pos = x_bottom_right_pos.reshape(self.wx*self.hx, self.dim)  

        z_seq_len = z_top_left_pos.shape[0]  
        x_seq_len = x_bottom_right_pos.shape[0]  

        z_pos_q = self.z_pos_q_linear(z_top_left_pos).view(z_seq_len, self.num_heads, -1).transpose(0, 1) * self.scale  
        z_pos_k = self.z_pos_k_linear(z_top_left_pos).view(z_seq_len, self.num_heads, -1).transpose(0, 1) 

        x_pos_q = self.x_pos_q_linear(x_bottom_right_pos).view(x_seq_len, self.num_heads, -1).transpose(0,
                                                                                                    1) * self.scale  
        x_pos_k = self.x_pos_k_linear(x_bottom_right_pos).view(x_seq_len, self.num_heads, -1).transpose(0,
                                                                                                    1)  
        return z_pos_q, z_pos_k, x_pos_q, x_pos_k



class UntiedPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_len, scale=None, with_q=True, with_k=True):
        super(UntiedPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = nn.Parameter(torch.empty(max_len, dim))
        trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        seq_len = self.pos.data.shape[0]
        pos = self.norm(self.pos)
        if self.pos_q_linear is not None and self.pos_k_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_q, pos_k
        elif self.pos_q_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            return pos_q
        elif self.pos_k_linear is not None:
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_k
        else:
            raise RuntimeError


