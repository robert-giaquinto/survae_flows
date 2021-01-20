import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module
from survae.nn.nets.conv import GatedConv, Conv2d
from survae.nn.layers import Rescale


#class TransformerNet(nn.Sequential):
class TransformerNet(nn.Module):
    """
    Neural network used to parametrize the transformations of a mixture
    of logistics coupling flow layer. This is the network that learns mixture
    parameters for the Flow++ model.

    An `MixLogisticAttnNet` is a stack of blocks, where each block consists of the following
    two layers connected in a residual fashion:
      1. Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
      2. Attn: input -> conv1x1 -> multihead self-attention -> gate,
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).

    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017), and the multi-head self attention mechanism we
    use is identical to the one in the Transformer (Vaswani et al., 2017).

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in each block of the network.
        num_blocks (int): Number of blocks in the network.
        num_mixtures (int): Number of components in the mixture.
        dropout (float): Dropout probability.
        use_attn (bool): Use attention in each block.
        context_channels (int): Number of channels in optional contextual auxillary input.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, num_mixtures, dropout, use_attn=True, context_channels=None, in_lambda=None, out_lambda=None):
        super(TransformerNet, self).__init__()
        
        self.in_lambda = LambdaLayer(in_lambda) if in_lambda else None
            
        self.conv1 = Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        attn_blocks = [ConvAttnBlock(mid_channels, dropout, use_attn, context_channels) for _ in range(num_blocks)]
        self.attn_blocks = nn.ModuleList(attn_blocks)
        
        self.conv2 = Conv2d(mid_channels, in_channels * (2 + 3 * num_mixtures), kernel_size=3, padding=1)
        self.out_lambda = layers.append(LambdaLayer(out_lambda)) if out_lambda else None

        # layers = []
        # if in_lambda: layers.append(LambdaLayer(in_lambda))
        # layers += [Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)]
        # layers += [ConvAttnBlock(mid_channels, dropout, use_attn, context_channels) for _ in range(num_blocks)]
        # layers += [Conv2d(mid_channels, in_channels * (2 + 3 * num_mixtures), kernel_size=3, padding=1)]
        # if out_lambda: layers.append(LambdaLayer(out_lambda))
        #super(TransformerNet, self).__init__(*layers)

    def forward(self, x, context=None):
        if self.in_lambda:
            x = self.in_lambda(x)

        x = self.conv1(x)
        for attn in self.attn_blocks:
            x = attn(x, context)
        x = self.conv2(x)

        if self.out_lambda:
            x = self.out_lambda(x)

        return x
        


class ConvAttnBlock(nn.Module):
    def __init__(self, channels, dropout, use_attn, context_channels):
        super(ConvAttnBlock, self).__init__()
        self.conv = GatedConv(channels, context_channels=context_channels, dropout=dropout)
        self.norm_1 = nn.LayerNorm(channels)
        if use_attn:
            self.attn = GatedAttn(channels, dropout=dropout)
            self.norm_2 = nn.LayerNorm(channels)
        else:
            self.attn = None

    def forward(self, x, context=None):
        x = self.conv(x, context) + x
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        x = self.norm_1(x)

        if self.attn:
            x = self.attn(x) + x
            x = self.norm_2(x)
        
        x = x.permute(0, 3, 1, 2)  # (b, c, h, w)

        return x


class GatedAttn(nn.Module):
    """
    Gated Multi-Head Self-Attention Block
    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.):
        super(GatedAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.in_proj = weight_norm(nn.Linear(d_model, 3 * d_model, bias=False))
        self.gate = weight_norm(nn.Linear(d_model, 2 * d_model))

    def forward(self, x):
        # Flatten and encode position
        b, h, w, c = x.size()
        x = x.view(b, h * w, c)
        _, seq_len, num_channels = x.size()
        pos_encoding = self.get_pos_enc(seq_len, num_channels, x.device)
        x = x + pos_encoding

        # Compute q, k, v
        memory, query = torch.split(self.in_proj(x), (2 * c, c), dim=-1)
        q = self.split_last_dim(query, self.num_heads)
        k, v = [self.split_last_dim(tensor, self.num_heads)
                for tensor in torch.split(memory, self.d_model, dim=2)]

        # Compute attention and reshape
        key_depth_per_head = self.d_model // self.num_heads
        q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(q, k, v)
        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))
        x = x.transpose(1, 2).view(b, c, h, w).permute(0, 2, 3, 1)  # (b, h, w, c)

        x = self.gate(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v, bias=False):
        """
        Dot-product attention.
        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.
        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            weights += self.bias
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(weights, self.dropout, self.training)
        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """
        Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.
        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)

        return ret

    @staticmethod
    def get_pos_enc(seq_len, num_channels, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        num_timescales = num_channels // 2
        log_timescale_increment = math.log(10000.) / (num_timescales - 1)
        inv_timescales = torch.arange(num_timescales,
                                      dtype=torch.float32,
                                      device=device)
        inv_timescales *= -log_timescale_increment
        inv_timescales = inv_timescales.exp_()
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        encoding = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
        encoding = F.pad(encoding, [0, num_channels % 2, 0, 0])
        encoding = encoding.view(1, seq_len, num_channels)

        return encoding