"""
ResidualBlock implements a block as described:

ResdualBlock (for middle block)
    for x                for time_emb             for x
    GroupNorm
    SiLU
    Conv k3p1s1          Linear
               add
    GroupNorm
    SiLU
    Conv k3p1s1                                  shortcut_conv k1p0s1
               add
               output

This block processes x as follows:
    - x goes through GroupNorm, SiLU, Conv3x3.
    - time_emb goes through SiLU, Linear and is added after Conv3x3.
    - then x goes through GroupNorm, SiLU, Conv3x3
    - possibly add a shortcut_conv 1x1 to x if shape changes.
    - finally add shortcut to x and return.

Assume:
    - nn.GroupNorm, nn.SiLU, nn.Conv2d, nn.Linear, and relevant torch objects exist in context.
    - self.in_channels, self.out_channels, self.time_emb_dim are set.
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_act = nn.SiLU()
        self.time_linear = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_h = self.time_act(time_emb)
        time_h = self.time_linear(time_h)
        # reshape for broadcast: [B, C] -> [B, C, 1, 1]
        h = h + time_h[..., None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        shortcut = self.shortcut(x)
        out = h + shortcut
        return out

class ResidualBlockUpDown(nn.Module):
    """
    Residual Block for up/down sampling, as described:
    norm -> acti -> [up/down sample] -> conv -> norm -> acti -> conv (+ shortcut)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False, down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()

        # Up or Down sample module
        if up:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = None
        if down:
            self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_act = nn.SiLU()
        self.time_linear = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            # Residual path needs channel projection
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # If upsampling or downsampling, shortcut needs to match spatial size
        if up:
            self.shortcut_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.shortcut_up = None
        if down:
            self.shortcut_down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut_down = None

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act1(h)
        # Up/down sampling logic after activation
        if self.up:
            h = self.upsample(h)
            shortcut = self.shortcut_up(x) if self.shortcut_up is not None else x
        elif self.down:
            h = self.downsample(h)
            shortcut = self.shortcut_down(x) if self.shortcut_down is not None else x
        else:
            shortcut = self.shortcut(x)
        h = self.conv1(h)

        time_h = self.time_act(time_emb)
        time_h = self.time_linear(time_h)
        h = h + time_h[..., None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        out = h + shortcut
        return out

        # INSERT_YOUR_CODE
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.inner_dim = channels  # 默认 值 相同

        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Linear(channels, self.inner_dim)
        self.k = nn.Linear(channels, self.inner_dim)
        self.v = nn.Linear(channels, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, channels)

    def forward(self, x):
        # x: (batch, channels, height, width)
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        x_flat = norm_x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        q = self.q(x_flat)
        k = self.k(x_flat)
        v = self.v(x_flat)

        # Reshape to (b, h*w, num_heads, head_dim)
        head_dim = self.inner_dim // self.num_heads
        q = q.view(b, h * w, self.num_heads, head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        k = k.view(b, h * w, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (b, num_heads, h*w, h*w)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (b, num_heads, h*w, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, h * w, self.inner_dim)  # (b, h*w, inner_dim)

        out = self.proj_out(attn_out)  # (b, h*w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
        return x + out

        # INSERT_YOUR_CODE
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, inner_dim, num_heads=8):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, q_dim)
        self.q_linear = nn.Linear(q_dim, inner_dim)
        self.k_linear = nn.Linear(kv_dim, inner_dim)
        self.v_linear = nn.Linear(kv_dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, q_dim)

    def forward(self, x, cross_x):
        # x: (bz, q_dim, h, w)
        # cross_x: (bz, kv_dim, emb_dim) or (bz, kv_dim)

        b, c, h, w = x.shape
        norm_x = self.norm(x)
        x_flat = norm_x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, q_dim)
        q = self.q_linear(x_flat)  # (b, h*w, inner_dim)

        if cross_x.dim() == 2:
            # If cross_x is (bz, kv_dim), treat as 1 token
            cross_x_flat = cross_x.unsqueeze(1)
        elif cross_x.dim() == 3:
            # If cross_x is (bz, kv_dim, emb_dim)
            cross_x_flat = cross_x
        else:
            raise ValueError("cross_x must be 2D or 3D")

        k = self.k_linear(cross_x_flat)  # (b, kv, inner_dim)
        v = self.v_linear(cross_x_flat)  # (b, kv, inner_dim)

        head_dim = self.inner_dim // self.num_heads
        q = q.view(b, h * w, self.num_heads, head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, head_dim).transpose(1, 2)     # (b, num_heads, kv, head_dim)
        v = v.view(b, -1, self.num_heads, head_dim).transpose(1, 2)     # (b, num_heads, kv, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (b, num_heads, h*w, kv)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (b, num_heads, h*w, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, h * w, self.inner_dim)  # (b, h*w, inner_dim)

        out = self.proj_out(attn_out)  # (b, h*w, q_dim)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)  # (b, q_dim, h, w)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, cross_c, temb_dim):
        super().__init__()
        self.res_block = ResidualBlockUpDown(in_c, out_c, temb_dim, down=True)
        self.self_attn = AttentionBlock(out_c)
        self.cross_attn = CrossAttentionBlock(out_c, cross_c, inner_dim=out_c)

    def forward(self, x, cross_x, temb):
        x = self.res_block(x, temb)
        x = self.self_attn(x)
        x = self.cross_attn(x, cross_x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, resdual_c, out_c, cross_c, temb_dim):
        super().__init__()
        self.res_block = ResidualBlockUpDown(in_c + resdual_c, out_c, temb_dim, up=True)
        self.self_attn = AttentionBlock(out_c)
        self.cross_attn = CrossAttentionBlock(out_c, cross_c, inner_dim=out_c)

    def forward(self, x, cross_x, temb, resdual):
        # x: (bz, in_c, h, w)
        # resdual: (bz, resdual_c, h, w)
        # cross_x: (bz, cross_c, emb_dim) or (bz, cross_c)
        # temb: (bz, temb_dim)
        x = torch.cat([x, resdual], dim=1)
        x = self.res_block(x, temb)
        x = self.self_attn(x)
        x = self.cross_attn(x, cross_x)
        return x


class MidBlock(nn.Module):
    def __init__(self, in_c, cross_c, temb_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_c, in_c, temb_dim)
        self.self_attn = AttentionBlock(in_c)
        self.cross_attn = CrossAttentionBlock(in_c, cross_c, inner_dim=in_c)

    def forward(self, x, cross_x, temb):
        x = self.res_block(x, temb)
        x = self.self_attn(x)
        x = self.cross_attn(x, cross_x)
        return x



class UNet(nn.Module):
    def __init__(self, in_c=4, out_c=4, base_c=96, cross_c=128, temb_dim=256):
        super().__init__()
        # Initial conv ("in_conv k3p1s1")
        self.in_conv = nn.Conv2d(in_c, base_c, kernel_size=3, padding=1, stride=1)
        # Down process: 3 levels
        self.down1 = DownBlock(base_c, 128, cross_c, temb_dim)
        self.down2 = DownBlock(128, 256, cross_c, temb_dim)
        self.down3 = DownBlock(256, 384, cross_c, temb_dim)
        # Mid process
        self.mid = MidBlock(384, cross_c, temb_dim)
        # Up process: 3 levels
        self.up1 = UpBlock(384, 384, 256, cross_c, temb_dim)
        self.up2 = UpBlock(256, 256, 128, cross_c, temb_dim)
        self.up3 = UpBlock(128, 128, base_c, cross_c, temb_dim)
        # Output groupnorm, SiLU, and final conv ("conv_out k3p1s1")
        self.out_norm = nn.GroupNorm(8, base_c)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_c, out_c, kernel_size=3, padding=1, stride=1)

    def forward(self, x, cross_x, temb):
        # Encoder
        x = self.in_conv(x) # torch.Size([2, 4, 64, 64]) -> torch.Size([2, 96, 64, 64])
        res_list = []
        x = self.down1(x, cross_x, temb); res_list.append(x) # torch.Size([2, 128, 32, 32])
        x = self.down2(x, cross_x, temb); res_list.append(x) # torch.Size([2, 256, 16, 16])
        x = self.down3(x, cross_x, temb); res_list.append(x) # torch.Size([2, 384, 8, 8])
        # Mid
        x = self.mid(x, cross_x, temb) # torch.Size([2, 384, 8, 8])
        # Decoder (reverse order, upsampling and residual connections)
        x = self.up1(x, cross_x, temb, res_list.pop()) # torch.Size([2, 256, 16, 16])
        x = self.up2(x, cross_x, temb, res_list.pop()) # torch.Size([2, 128, 32, 32])
        x = self.up3(x, cross_x, temb, res_list.pop()) # torch.Size([2, 96, 64, 64])
        # Output
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x) # torch.Size([2, 4, 64, 64])
        return x

def test_UNet():
    import argparse,os
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=str, default='2', help='CUDA device id')
    args, unknown = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    input = torch.randn((2,4,64,64)).cuda()
    temb = torch.randn((2,256)).cuda()
    cross_input = torch.randn((2,77,768)).cuda()

    in_c = 4
    out_c = 4
    base_c = 96
    cross_c = 768
    temb_dim = 256

    unet = UNet(
        in_c=in_c,
        out_c=out_c,
        base_c=base_c,
        cross_c=cross_c,
        temb_dim=temb_dim,
    ).to('cuda:0')

    unet(
        x = input,
        cross_x = cross_input,
        temb = temb,
    )


    # time_emb = in_c * 4

if __name__=='__main__':
    test_UNet()