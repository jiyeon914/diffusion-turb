import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_gan_3d.models.encoder import Encoder
from vq_gan_3d.models.decoder import Decoder
from vq_gan_3d.models.codebook import Codebook
from vq_gan_3d.util import shift_dim, adopt_weight



def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    # d_loss = 0.5 * (loss_real + loss_fake)
    return loss_real, loss_fake

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.softplus(-logits_real))
    loss_fake = torch.mean(torch.nn.functional.softplus(logits_fake))
    # d_loss = 0.5 * (loss_real + loss_fake)
    return loss_real, loss_fake

class VQGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = Encoder(cfg.model.dim, cfg.model.init_dim, cfg.model.out_dim, cfg.model.dim_mults, cfg.model.channels,
            cfg.model.z_channels, cfg.model.resnet_block_groups)
        self.decoder = Decoder(cfg.model.dim, cfg.model.init_dim, cfg.model.out_dim, cfg.model.dim_mults, cfg.model.channels,
            cfg.model.z_channels, cfg.model.resnet_block_groups)

        self.enc_z_ch = self.encoder.z_channels
        self.embed_dim = cfg.model.embed_dim if cfg.model.embed_dim is not None else self.enc_z_ch
        self.pre_vq_conv = nn.Conv3d(self.enc_z_ch, self.embed_dim, 1)
        self.post_vq_conv = nn.Conv3d(self.embed_dim, self.enc_z_ch, 1)
        self.codebook = Codebook(cfg.model.n_codes, self.enc_z_ch)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.discriminator = NLayerDiscriminator3D(cfg.model.channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm3d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        self.disc_iter_start = cfg.model.disc_iter_start

        self.gan_weight = cfg.model.gan_weight
        self.l1_weight = cfg.model.l1_weight
        self.global_step = 0

    def encode(self, x, include_embeddings=False, quantize=True):
        z = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(z)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return z

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        return x_recon, vq_output
    
    def compute_loss(self, x, optimizer_idx):
        B, C, D, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"
            # Discriminator loss (turned on after a certain epoch)
            logits_fake, pred_fake = self.discriminator(x_recon)
            g_loss = -self.gan_weight*torch.mean(logits_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.disc_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.gan_weight > 0:
                logits_real, pred_real = self.discriminator(x)
                for i in range(len(pred_fake)-1):
                    gan_feat_loss += feat_weights * F.l1_loss(pred_fake[i], pred_real[i].detach()) * (self.gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * gan_feat_loss
            return recon_loss, vq_output, aeloss, gan_feat_loss #perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            logits_real, _ = self.discriminator(x.detach())
            logits_fake, _ = self.discriminator(x_recon.detach())

            loss_real, loss_fake = self.disc_loss(logits_real, logits_fake)
            d_loss = 0.5 * (loss_real + loss_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.disc_iter_start)

            discloss = disc_factor * (self.gan_weight*d_loss)
            return discloss, loss_real, loss_fake
        
    def update_step(self):
        # increase global_step every iteration
        self.global_step += 1

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _
