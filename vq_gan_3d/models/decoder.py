from functools import partial

from torch import nn
import torch.nn.functional as F

# from .modules import default, Residual, PreNorm, Upsample, ResnetBlock, Attention, LinearAttention, make_attn
from vq_gan_3d.models.module import default, Upsample, ResnetBlock



class Decoder(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
                 z_channels=None, resnet_block_groups=32):
        super().__init__()

        # determine dimensions
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_blocks = len(in_out)

        # layers
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        mid_channels = dims[-1]
        self.z_channels = z_channels if z_channels is not None else mid_channels
        self.conv_in = nn.Conv3d(self.z_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.mid_block = block_klass(mid_channels, mid_channels)
        
        self.ups = nn.ModuleList([])
        for idx, (in_channels, out_channels) in enumerate(reversed(in_out)):
            is_last = idx == (num_blocks - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                    block_klass(out_channels, out_channels),
                    block_klass(out_channels, out_channels),
                    Upsample(out_channels, in_channels) if not is_last
                    else nn.Conv3d(out_channels, in_channels, kernel_size=3, padding=1, padding_mode='circular'),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_conv = nn.Conv3d(dims[0], self.out_dim, 1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block1, block2, upsample in self.ups:
            h = block1(h)
            h = block2(h)
            h = upsample(h)
        return self.final_conv(h)


# class Decoder(nn.Module):
#     def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
#                  z_channels=None, resnet_block_groups=32, attn_type="vanilla"):
#         super().__init__()

#         # determine dimensions
#         init_dim = default(init_dim, dim)
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#         num_blocks = len(in_out)

#         # layers
#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)

#         mid_channels = dims[-1]
#         self.z_channels = z_channels if z_channels is not None else mid_channels
#         self.mid_block1 = block_klass(self.z_channels, mid_channels)
#         self.mid_attn = Residual(PreNorm(mid_channels, Attention(mid_channels)))
#         self.mid_block2 = block_klass(mid_channels, mid_channels)

#         self.ups = nn.ModuleList([])
#         for idx, (in_channels, out_channels) in enumerate(reversed(in_out)):
#             is_last = idx == (num_blocks - 1)
#             self.ups.append(
#                 nn.ModuleList([
#                     block_klass(out_channels, out_channels),
#                     block_klass(out_channels, out_channels),
#                     Residual(PreNorm(out_channels, make_attn(out_channels, attn_type=attn_type))),
#                     Upsample(out_channels, in_channels) if not is_last
#                     else nn.Conv3d(out_channels, in_channels, kernel_size=3, padding=1),])
#             )

#         self.out_dim = default(out_dim, in_channels)
#         self.final_res_block = block_klass(dims[0], dim)
#         self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

#     def forward(self, z):
#         h = self.mid_block1(z)
#         h = self.mid_attn(h)
#         h = self.mid_block2(h)

#         for block1, block2, attn, upsample in self.ups:
#             h = block1(h)
#             h = block2(h)
#             h = attn(h)
#             h = upsample(h)

#         h = self.final_res_block(h)
#         return self.final_conv(h)
