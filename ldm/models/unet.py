from functools import partial

import torch
import torch.nn as nn

from ldm.models.module import default, Residual, PreNorm, Downsample, Upsample, ResnetBlock, Attention, LinearAttention, make_attn, Positional_Embedding



class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # determine dimensions
        self.channels = cfg.unet.channels
        self.self_condition = cfg.unet.self_condition
        # self.cond_dim = cond_dim #if cond_dim_type is '' else
        input_channels = self.channels * (2 if self.self_condition else 1) #channels + (self.cond_dim if self_condition and cond_dim is not None else 0)

        init_dim = default(cfg.unet.init_dim, cfg.unet.dim)
        self.init_conv = nn.Conv3d(in_channels=input_channels, out_channels=init_dim, kernel_size=1, padding=0) # changed to 1 and 0 from 7 and 3

        dims = [init_dim, *map(lambda m: cfg.unet.dim * m, cfg.unet.dim_mults)] # (64, 128, 256, 512, 1024) when dim = 128, init_dim = 64, dim_mults = (1, 2, 4, 8)
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = cfg.unet.dim * 4
        self.time_mlp = nn.Sequential(
            Positional_Embedding(cfg.unet.dim),
            nn.Linear(cfg.unet.dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_block = len(in_out)

        block_klass = partial(ResnetBlock, groups=cfg.unet.resnet_block_groups)
        for idx, (in_channels, out_channels) in enumerate(in_out):
            is_last = idx >= (num_block - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                    block_klass(in_channels, in_channels, time_emb_dim=time_dim),
                    block_klass(in_channels, in_channels, time_emb_dim=time_dim),
                    Residual(PreNorm(in_channels, LinearAttention(in_channels))),
                    Downsample(in_channels, out_channels) if not is_last
                    else nn.Conv3d(in_channels, out_channels, 3, padding=1),
                    ]
                )
            )

        mid_channels = dims[-1]
        self.mid_block1 = block_klass(mid_channels, mid_channels, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_channels, Attention(mid_channels)))
        self.mid_block2 = block_klass(mid_channels, mid_channels, time_emb_dim=time_dim)

        for idx, (in_channels, out_channels) in enumerate(reversed(in_out)):
            is_last = idx >= (num_block - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                    block_klass(out_channels + in_channels, out_channels, time_emb_dim=time_dim),
                    block_klass(out_channels + in_channels, out_channels, time_emb_dim=time_dim),
                    Residual(PreNorm(out_channels, LinearAttention(out_channels))),
                    Upsample(out_channels, in_channels) if not is_last
                    else nn.Conv3d(out_channels, in_channels, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(cfg.unet.out_dim, cfg.unet.channels)
        self.final_res_block = block_klass(dims[0] * 2, cfg.unet.dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(cfg.unet.dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, x_self_cond), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)