# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention
from ldm.modules.swinir import SpatialAwareWindowAttention
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
from basicsr.archs.rrdbnet_arch import RRDB
from ldm.modules.swinir import SwinTransformerBlock, PatchEmbed, PatchUnEmbed

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class SpatialAwarePixelAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, mask=None):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        # compute attention mask 
        if mask is not None:
            mask = mask.float()
            mask = mask.squeeze(0)
            height = mask.shape[0]
            width = mask.shape[1]
            mask = mask.reshape(1, height, width)
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=(h,w))
            #mask = mask.squeeze(0)
            mask_1 = mask.view(h*w, 1)
            mask_2 = mask.view(1,h*w)
            mask = torch.matmul(mask_1, mask_2)
            mask = mask.unsqueeze(0)
            #print(w_.shape)

            # mask = mask.repeat(1, 1, h*w)
            
            

            # w_ = w_.masked_fill(mask == 0, -1e9)
        
        
        #print(mask)
        h_sub = w_
        #print(w_)
        C = h*w
        mask1 = torch.zeros(b, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, C, C, device=x.device, requires_grad=False)
        # if mask is not None:
        reverse_mask = 1 - mask
        # w_ = w_ * reverse_mask
        #     w_ = w_.permute(0,2,1)
            
        index = torch.topk(mask, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, w_, torch.full_like(w_, -100))

        index = torch.topk(mask, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, w_, torch.full_like(w_, -100))

        index = torch.topk(mask, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, w_, torch.full_like(w_, -100))

        index = torch.topk(mask, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, w_, torch.full_like(w_, -100))
        #w_ = torch.nn.functional.softmax(w_, dim=2)
        h_sub = attn1
        attn1 = torch.nn.functional.softmax(attn1, dim=-1)
        attn2 = torch.nn.functional.softmax(attn2, dim=-1)
        attn3 = torch.nn.functional.softmax(attn3, dim=-1)
        attn4 = torch.nn.functional.softmax(attn4, dim=-1)

        v = v.reshape(b,c,h*w)

        out1 = torch.bmm(v, attn1)
        out2 = torch.bmm(v, attn2)
        out3 = torch.bmm(v, attn3)
        out4 = torch.bmm(v, attn4)
        


        h_ = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
            

        #     h_ = h_.reshape(b,c,h,w)
            #h_sub = h_sub.sum(dim=1)

            #h_sub = h_sub.reshape(b,1,h*w1,w)
            #h_ = self.proj_out(h_)
            
            #h_sub = torch.clamp(h_sub, 0, 1)
            #h_sub = (h_sub-torch.min(h_sub))/(torch.max(h_sub)-torch.min(h_sub))
            #print(h_sub)
        # w_ = (w_-torch.min(w_))/(torch.max(w_)-torch.min(w_))
        # attend to values
        #else: 
        #w_ = attn1
       # w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        
        #w_ = torch.nn.functional.softmax(w_, dim=2)
        
        #h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)
        #h_sub = h_sub.reshape(b,1,h,w)
        #h_sub = torch.nn.functional.softmax(h_sub, dim=2)
        h_ = self.proj_out(h_)
        
        #h_sub = (h_sub-torch.min(h_sub))/(torch.max(h_sub)-torch.min(h_sub))
            
        
        return x+h_, h_sub
        # x+h_, h_sub


def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

class MemoryEfficientAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q, k, v = map(
            lambda t:t.reshape(b, t.shape[1], t.shape[2]*t.shape[3], 1)
            .squeeze(3)
            .permute(0,2,1)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, scale=(int(c)**(-0.5)), op=self.attention_op)

        h_ = (
            out.permute(0,2,1)
            .unsqueeze(3)
            .reshape(b, c, h, w)
        )

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", "spatial"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        if XFORMERS_IS_AVAILBLE:
            return MemoryEfficientAttnBlock(in_channels)
        else:
            return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == "spatial":
        print("using_spatial_aware_attn")
        return SpatialAwareWindowAttention(dim=in_channels, window_size=16, num_heads=1)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, use_spatial_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        if use_spatial_attn: attn_type = "spatial"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2   #x01.shape=[4,3,128,256]   从0开始，每隔两个取出，#像素值还要除以2    0,2,4,6...254-->0,1,2,...127
    x02 = x[:, :, 1::2, :] / 2   #x02.shape=[4,3,128,256]   从1开始，每隔两个取出，#像素值还要除以2    1,3,5,7...255-->0,1,2,...127 
    x1 = x01[:, :, :, 0::2]    #x1.shape=[4,3,128,128]   从0取出      0,2,4,6...254-->0,1,2,...127
    x2 = x02[:, :, :, 0::2]       #x2.shape=[4,3,128,128]   从0取出
    x3 = x01[:, :, :, 1::2]     #x3.shape=[4,3,128,128]   从1取出     1,3,5,7...255-->0,1,2,...127 
    x4 = x02[:, :, :, 1::2]  #x4.shape=[4,3,128,128]   从1取出
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)

class DWT_transform(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency,dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency,dwt_high_frequency

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, use_spatial_attn=True, attn_type="vanilla",
                **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        if use_spatial_attn: attn_type = "spatial"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)

        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_spatial_attn = use_spatial_attn
        self.norm = nn.LayerNorm(256)

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions): # num_resolution = 4
            block = nn.ModuleList()
            attn = nn.ModuleList()
            patch_embed = nn.ModuleList()
            patch_unembed = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SwinTransformerBlock(dim=block_in, 
                                                     input_resolution=curr_res, 
                                                     num_heads=1,
                                                     window_size=16))
            #         patch_embed.append(PatchEmbed(
            # img_size=512, patch_size=64, in_chans=block_in, embed_dim=block_in,
            # norm_layer=nn.LayerNorm))
            #         patch_unembed.append(PatchUnEmbed(
            # img_size=512, patch_size=64, in_chans=block_in, embed_dim=block_in,
            # norm_layer=nn.LayerNorm))

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.patch_embed = patch_embed
            down.patch_unembed = patch_unembed
            down.dwt = DWT_transform(block_in, block_out)
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                
                curr_res = curr_res // 2
            # merge = torch.nn.Conv2d(2*block_in, block_in, kernel_size=3, padding=1).cuda()
            # down.merge = merge
            self.down.append(down)
            

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.patch_embed = PatchEmbed(
        #     img_size=512, patch_size=64, in_chans=block_in, embed_dim=512,
        #     norm_layer=nn.LayerNorm)
        self.mid.attn_1 = SwinTransformerBlock(dim=block_in, 
                                                     input_resolution=curr_res, 
                                                     num_heads=1,
                                                     window_size=16)
        # self.mid.patch_unembed = PatchUnEmbed(
        #     img_size=512, patch_size=64, in_chans=block_in, embed_dim=512,
        #     norm_layer=nn.LayerNorm)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_2 = make_attn(block_in, attn_type="vanilla")
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, mask=None, return_fea=False, return_high_freq=True):
        # timestep embedding
        device = x.device
        x = x.float()
        
        if self.use_spatial_attn == False:
            mask = None
        temb = None
        if return_high_freq == False: 
            # downsampling
            hs = [self.conv_in(x)]
            fea_list = []
            for i_level in range(self.num_resolutions): # i_level = 0,1,2,3
                for i_block in range(self.num_res_blocks): # resblocks : 2
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    if len(self.down[i_level].attn) > 0:
                        h, attention_map = self.down[i_level].attn[i_block](h)
                        #h = self.down[i_level].attn[i_block](h)
                    hs.append(h)
                if return_fea:
                    if i_level==1 or i_level==2:
                        fea_list.append(h)
                if i_level != self.num_resolutions-1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
            
            # middle
            h = hs[-1]
            
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_2(h)
            # h = self.mid.attn_1(h)
            
            h = self.mid.block_2(h, temb)
            


            # end
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
            

            if return_fea:
                return h, fea_list
            return h

        else:
            hs = [self.conv_in(x)]
            fea_list = []
            attn_list = []
            high_freq_fea = []
            for i_level in range(self.num_resolutions): # i_level = 0,1,2,3
                for i_block in range(self.num_res_blocks): # resblocks : 2
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    if len(self.down[i_level].attn) > 0:
                        h_size = (h.shape[2], h.shape[3])
                        
                        mask = mask.float()
                        mask = mask.unsqueeze(0)
                        
                        mask = torch.nn.functional.interpolate(mask, size=(h.shape[2], h.shape[3]))
                        mask = torch.clamp(mask, min=0, max=1)
                        # print("mask")
                        # print(mask)
                        _,C,_,_ = h.shape
                        mask_ = mask.repeat(1, C, 1, 1)

                        # h = torch.cat([(1-mask_),h], dim=1)
                        # h = torch.cat([mask_,h], dim=1)
                        mask = mask.squeeze(0)
                        # h = self.down[i_level].patch_embed[i_block](h)
                        
                        reverse_mask = 1 - mask

                        h = self.down[i_level].attn[i_block](h, h_size, reverse_mask)
                        # h = self.down[i_level].patch_unembed[i_block](h, h_size)
                        
                        #h = h[:,0:C,:,:]
                        #print(h.shape)
                        dwt_low, dwt_high = self.down[i_level].dwt(h)
                        high_freq_fea.append(dwt_low)
                        #h = self.down[i_level].attn[i_block](h)
                    hs.append(h)
                if return_fea:
                    if i_level==1 or i_level==2:
                        fea_list.append(h)
                if i_level != self.num_resolutions-1:
                    hs_ = hs[-1]
                    hs_down = self.down[i_level].downsample(hs[-1])
                    
                    hs[-1].squeeze(0)
                    #hs_out = torch.cat([hs_down, dwt_low],dim=1)
                    hs_out = hs_down # + dwt_low
                    #hs_out = self.down[i_level].merge(hs_out)
                    hs.append(hs_out)
                    

                

            # middle
            h = hs[-1]
            h = self.mid.block_1(h, temb)
            h_size = (h.shape[2], h.shape[3])
            _,C,_,_ = h.shape
            mask = mask.float()
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=(h.shape[2], h.shape[3]))
            mask_ = mask.repeat(1, C, 1, 1)
            # h = torch.cat([(1-mask_), h], dim=1)
            # h = torch.cat([mask_,h], dim=1)
            # h = self.mid.patch_embed(h)
            mask = mask.squeeze(0)
            reverse_mask = 1 - mask

            h = self.mid.attn_1(h, h_size, reverse_mask)
            # h = self.mid.patch_unembed(h, h_size)
           # h = h[:,0:C,:,:]
            # h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)


            # end
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)

            if return_fea:
                return h, fea_list, high_freq_fea, attn_list
            return h, high_freq_fea, attn_list
        

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, use_spatial_attn=True,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class Decoder_Mix(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, use_spatial_attn=True,
                 attn_type="vanilla", num_fuse_block=2, fusion_w=1.0, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        if use_spatial_attn: attn_type = "spatial"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.fusion_w = fusion_w
        self.use_spatial_attn = use_spatial_attn

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))
        

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.patch_embed = PatchEmbed(
        #     img_size=512, patch_size=64, in_chans=block_in, embed_dim=512,
        #     norm_layer=nn.LayerNorm)
        self.mid.attn_1 = SwinTransformerBlock(dim=block_in, 
                                                     input_resolution=curr_res, 
                                                     num_heads=1,
                                                     window_size=16)
        # self.mid.patch_unembed = PatchUnEmbed(
        #     img_size=512, patch_size=64, in_chans=block_in, embed_dim=512,
        #     norm_layer=nn.LayerNorm)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_2 = make_attn(block_in,attn_type="vanilla")

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # i_level = 3,2,1,0
            block = nn.ModuleList()
            attn = nn.ModuleList()
            patch_embed = nn.ModuleList()
            patch_unembed = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            

            if i_level != self.num_resolutions-1: # i_level != 3
                if i_level != 0:
                    fuse_layer = Fuse_sft_block_RRDB(in_ch=block_out, out_ch=block_out, num_block=num_fuse_block)
                    setattr(self, 'fusion_layer_{}'.format(i_level), fuse_layer)
                    # set fusion_layers

            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SwinTransformerBlock(dim=block_in, 
                                                     input_resolution=curr_res, 
                                                     num_heads=1,
                                                     window_size=16))
            #         patch_embed.append(PatchEmbed(
            # img_size=512, patch_size=64, in_chans=block_in, embed_dim=block_in,
            # norm_layer=nn.LayerNorm))
            #         patch_unembed.append(PatchUnEmbed(
            # img_size=512, patch_size=64, in_chans=block_in, embed_dim=block_in,
            # norm_layer=nn.LayerNorm))
            up = nn.Module()
            up.block = block
            up.attn = attn
            up.patch_embed = patch_embed
            up.patch_unembed = patch_unembed
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            # if i_level != self.num_resolutions-1 and i_level != 0:
            #     merge = torch.nn.Conv2d(int(block_out*1.5), block_out, kernel_size=3, padding=1).cuda()
            # else:
            #     merge = torch.nn.Conv2d(int(block_out*2), block_out,kernel_size=3, padding=1).cuda()
            # up.merge = merge
            
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, enc_fea, high_freq_fea, freq_merge=True, mask=None):
        if self.use_spatial_attn == False:
            mask = None
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        if freq_merge == False: 
            # z to block_in
            h = self.conv_in(z)

            # middle
            h = self.mid.block_1(h, temb)
            # h = self.mid.attn_1(h)
            h,_ = self.mid.attn_2(h)
            h = self.mid.block_2(h, temb)

            # upsampling
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks+1): # num_res_blocks+1= 3
                    h = self.up[i_level].block[i_block](h, temb)
                    if len(self.up[i_level].attn) > 0:
                        h,_ = self.up[i_level].attn[i_block](h)
                        # h = self.up[i_level].attn[i_block](h)

                if i_level != self.num_resolutions-1 and i_level != 0:
                    # i_level = 2, 1
                    cur_fuse_layer = getattr(self, 'fusion_layer_{}'.format(i_level))
                    h = cur_fuse_layer(enc_fea[i_level-1], h, self.fusion_w)

                if i_level != 0:
                    
                    h = self.up[i_level].upsample(h)
            # end
            if self.give_pre_end:
                return h

            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
            if self.tanh_out:
                h = torch.tanh(h)
            return h

        if freq_merge == True:
            # z to block_in
            h = self.conv_in(z)

            # middle
            h = self.mid.block_1(h, temb)
            h_size = (h.shape[2], h.shape[3])
            _,C,_,_ = h.shape
            
            
            mask = mask.float()
            mask = mask.unsqueeze(0)
            # print('decoder')
            # print(mask.shape)
            mask = torch.nn.functional.interpolate(mask, size=(h.shape[2], h.shape[3]))

            mask_ = mask.repeat(1, C, 1, 1)
            # h = torch.cat([(1-mask_), h], dim=1)
            # h = torch.cat([mask_, h], dim=1)
            # h = self.mid.patch_embed(h)
            mask = mask.squeeze(0)
            reverse_mask = 1 - mask

            h = self.mid.attn_1(h, h_size, reverse_mask)
            # h = self.mid.patch_unembed(h, h_size)
            
            h = self.mid.block_2(h, temb)

            # upsampling
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks+1): # num_res_blocks+1= 3
                    h = self.up[i_level].block[i_block](h, temb)
                    if len(self.up[i_level].attn) > 0:
                        h_size = (h.shape[2], h.shape[3])
                        _,C,_,_ = h.shape
                        mask = mask.float()
                        mask = mask.unsqueeze(0)
                        mask = torch.nn.functional.interpolate(mask, size=(h.shape[2], h.shape[3]))
                        mask_ = mask.repeat(1, C, 1, 1)
                        # h = torch.cat([(1-mask_), h], dim=1)
                        # h = torch.cat([mask_,h], dim=1)
                        # h = self.up[i_level].patch_embed[i_block](h)
                        
                        mask = mask.squeeze(0)
                        reverse_mask = 1 - mask

                        h = self.up[i_level].attn[i_block](h, h_size, reverse_mask)
                        # h = self.up[i_level].patch_unembed[i_block](h, h_size)
                        #h = h[:, 0:C, :,:]
                        # h = self.up[i_level].attn[i_block](h)
                # if i_level != 0:
                #     h = torch.cat([high_freq_fea[i_level-1], h], dim=1)
                #     h = self.up[i_level].merge(h)
                #     h = high_freq_fea[i_level-1] + h
                if i_level != self.num_resolutions-1 and i_level != 0:
                    # i_level = 2, 1
                    #print(i_level)
                    cur_fuse_layer = getattr(self, 'fusion_layer_{}'.format(i_level))
                    h = cur_fuse_layer(enc_fea[i_level-1], h, self.fusion_w)
                    
                    

                if i_level != 0:
                    
                    # _, c, _, _ = high_freq_fea[i_level-1].size()
                    # h_template = torch.zeros_like(h)
                    # h_template[:,0:c,:,:] = high_freq_fea[i_level-1]
                    # h = h_template + h
                    # del h_template
                    h = self.up[i_level].upsample(h)
                
            # end
            if self.give_pre_end:
                return h

            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
            if self.tanh_out:
                h = torch.tanh(h)
            return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nonlinearity(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class Fuse_sft_block_RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, num_grow_ch=32):
        super().__init__()
        self.encode_enc_1 = ResBlock(2*in_ch, in_ch)
        self.encode_enc_2 = make_layer(RRDB, num_block, num_feat=in_ch, num_grow_ch=num_grow_ch)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        residual = w * enc_feat
        out = dec_feat + residual
        return out

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        if XFORMERS_IS_AVAILBLE:
            self.attn = MemoryEfficientAttnBlock(mid_channels)
        else:
            self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z
