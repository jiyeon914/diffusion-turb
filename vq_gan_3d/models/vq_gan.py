import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .encoder import Encoder
from .decoder import Decoder
from .codebook import Codebook
from .lpips import LPIPS
from ..utils import shift_dim, adopt_weight



def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQGAN(nn.Module):
    def __init__(self, *, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
                 z_channels=None, resnet_block_groups=32, n_codes=2**13, embed_dim=None,
                 gan_feat_weight=4.0, disc_channels=64, disc_layers=3, disc_loss_type='vanilla', disc_iter_start=20000,
                 image_gan_weight=1.0, video_gan_weight=1.0, perceptual_weight=4.0, l1_weight=4.0):
        super().__init__()
        
        self.encoder = Encoder(dim, init_dim, out_dim, dim_mults, channels,
            z_channels, resnet_block_groups)
        self.decoder = Decoder(dim, init_dim, out_dim, dim_mults, channels,
            z_channels, resnet_block_groups)

        self.enc_z_ch = self.encoder.z_channels
        self.embed_dim = embed_dim if embed_dim is not None else self.enc_z_ch
        self.pre_vq_conv = nn.Conv3d(self.enc_z_ch, self.embed_dim, 1)
        self.post_vq_conv = nn.Conv3d(self.embed_dim, self.enc_z_ch, 1)
        self.codebook = Codebook(n_codes, self.enc_z_ch)

        self.gan_feat_weight = gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        # self.image_discriminator = NLayerDiscriminator(channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm3d)

        if disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        self.disc_iter_start = disc_iter_start

        self.perceptual_model = LPIPS().eval()

        # self.image_gan_weight = image_gan_weight
        self.video_gan_weight = video_gan_weight
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
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

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, D, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"
            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            # logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            # g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.video_gan_weight*g_video_loss# + self.image_gan_weight*g_image_loss
            disc_factor = adopt_weight(self.global_step, threshold=self.disc_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            # image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            # if self.image_gan_weight > 0:
            #     logits_image_real, pred_image_real = self.image_discriminator(frames)
            #     for i in range(len(pred_image_fake)-1):
            #         image_gan_feat_loss += feat_weights * \
            #             F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * video_gan_feat_loss#(image_gan_feat_loss + video_gan_feat_loss)
            return recon_loss, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            # logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            # logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            # d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.disc_iter_start)

            discloss = disc_factor * (self.video_gan_weight*d_video_loss)# + self.image_gan_weight*d_image_loss)
            return discloss
        
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
