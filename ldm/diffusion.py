import os
import math
pi = math.pi
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")

import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

from vq_gan_3d.models.vq_gan import VQGAN
from ldm.models.module import default



class beta_schedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def cosine_beta_schedule(self, s=0.008):
        """    cosine schedule as proposed in https://arxiv.org/abs/2102.09672    """
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps)
        f = torch.cos((t/self.timesteps + s)/(1 + s)*pi*0.5)**2
        alpha_bar = f/f[0]
        betas = 1 - (alpha_bar[1:]/alpha_bar[:-1])
        return torch.clip(betas, self.beta_start, 0.9999)

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def quadratic_beta_schedule(self):
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.timesteps)**2

    def sigmoid_beta_schedule(self):
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas)*(self.beta_end - self.beta_start) + self.beta_start

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class Diffusion:
    def __init__(self, denoise_fn, cfg):
        super().__init__()
        self.device = cfg.device
        if cfg.diffusion.vqgan_ckpt:
            vqgan_cfg_path = cfg.diffusion.vqgan_config
            with open(vqgan_cfg_path, 'r') as file:
                vqgan_cfg = yaml.safe_load(file)
                vqgan_cfg = OmegaConf.create(vqgan_cfg)
            self.vqgan = VQGAN(vqgan_cfg).to(self.device)

            vqgan_path = os.path.join(cfg.paths.file_dir, "MODELS", cfg.diffusion.vqgan_model_case, f"{cfg.diffusion.vqgan_ckpt}ckpt.pt")
            self.vqgan.load_state_dict(torch.load(vqgan_path))
            self.vqgan.eval()
        else:
            self.vqgan = None
        self.img_size = (cfg.data.data_res//2**(len(vqgan_cfg.model.dim_mults) - 1)
                         if self.vqgan is not None else cfg.data.data_res)
        self.channels = (self.vqgan.encoder.z_channels
                         if self.vqgan is not None else cfg.diffusion.channels)
        self.denoise_fn = denoise_fn
        self.loss_type = cfg.diffusion.loss_type
        
        
        schedule = beta_schedule(timesteps=cfg.diffusion.timesteps)
        self.timesteps = schedule.timesteps
        self.betas = schedule.cosine_beta_schedule().to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alpha_bar = 1. - self.alpha_bar
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.)
        self.one_minus_alpha_bar_prev = 1. - self.alpha_bar_prev
        self.beta_tilde = (1. - self.alpha_bar_prev)/(1. - self.alpha_bar)*self.betas
#         self.loss_weight = self.betas**2/(2*self.beta_tilde*self.alphas*(1. - self.alpha_bar));
#         self.loss_weight[-10:] = 0.1
             
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            1/torch.sqrt(extract(self.alpha_bar, t, x_t.shape)) * 
            (x_t - torch.sqrt(extract(self.one_minus_alpha_bar, t, x_t.shape)) * noise)
        )

    def q_posterior(self, x0, x_t, noise, t, sample_mode):
        if sample_mode == 'xt_x0':
            mean_coeff1 = (1/extract(self.one_minus_alpha_bar, t, x_t.shape) * 
                        torch.sqrt(extract(self.alphas, t, x_t.shape)) * 
                        extract(self.one_minus_alpha_bar_prev, t, x_t.shape)
            ) # coefficient of x_t
            mean_coeff2 = (1/extract(self.one_minus_alpha_bar, t, x_t.shape) * 
                        torch.sqrt(extract(self.alpha_bar_prev, t, x_t.shape)) * 
                        extract(self.betas, t, x_t.shape)
            ) # coefficient of x0
            posterior_mean = mean_coeff1*x_t + mean_coeff2*x0
        elif sample_mode == 'x0_eps':
            noise_coeff = (extract(self.one_minus_alpha_bar_prev, t, x_t.shape) * 
                           torch.sqrt(extract(self.alphas, t, x_t.shape) / 
                                      extract(self.one_minus_alpha_bar, t, x_t.shape))
            )
            posterior_mean = torch.sqrt(extract(self.alpha_bar_prev, t, x_t.shape))*x0 + noise_coeff*noise
        elif sample_mode == 'xt_eps':
            posterior_mean = (1/torch.sqrt(extract(self.alphas, t, x_t.shape)) * 
                              (x_t - extract(self.betas, t, x_t.shape) / 
                               torch.sqrt(extract(self.one_minus_alpha_bar, t, x_t.shape)) * noise)
            )
        else:
            raise NotImplementedError()
        
        posterior_variance = extract(self.beta_tilde, t, x_t.shape)
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, x_t, t, condition, clip_denoised: bool, sample_mode):
        noise = self.denoise_fn(x_t, t, x_self_cond=condition)
        x0_hat = self.predict_start_from_noise(x_t, t, noise)

        if clip_denoised:
            s = 1.
            if cfg.diffusion.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x0_hat, 'b ... -> b (...)').abs(),
                    cfg.diffusion.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x0_hat.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x0_hat = x0_hat.clamp(-s, s) / s

        model_mean, posterior_variance = self.q_posterior(x0=x0_hat, x_t=x_t, noise=noise, t=t, sample_mode=sample_mode)
        return model_mean, posterior_variance

    @torch.inference_mode()
    def p_sample(self, x_t, t_index, condition, clip_denoised=True, sample_mode='xt_x0'):
        b = x_t.shape[0]
        t = (torch.ones(n) * t_index).long().to(self.device)

        model_mean, model_variance = self.p_mean_variance(
            x_t=x_t, t=t, condition=condition, clip_denoised=clip_denoised, sample_mode=sample_mode)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * model_variance * noise

    @torch.inference_mode()
    def p_sample_loop(self, x_t, t_index, condition, sample_mode):
        b = x_t.shape[0]
        logging.info(f"Sampling {b} new images using {sample_mode} starting from t = {t_index}....")

        # loop = []
        for i in reversed(range(0, t_index)):
            x_t = self.p_sample(x_t, i, condition, sample_mode)
        #     loop.append(x_t)
        # x_t = torch.cat([loop[i] for i in range(t_index)], dim=1)
        return x_t

    @torch.inference_mode()
    def sample(self, x_t, t_index, condition, sample_mode='xt_x0'):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        _sample = self.p_sample_loop(x_t, t_index, condition, sample_mode)

        if isinstance(self.vqgan, VQGAN):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, device = x1.shape[0], x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched)[0], (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img
    
    def sample_timesteps(self, batch_size):
        # return torch.randint(low = 1, high = self.timesteps, size = (batch_size,))
        return torch.randint(low=0, high=self.timesteps, size=(batch_size,))
    
    def q_mean_variance(self, x0, t):
        mean = torch.sqrt(extract(self.alpha_bar, t, x0.shape)) * x0
        variance = extract(self.one_minus_alpha_bar, t, x0.shape)
        # log_variance = extract(
        #     self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance#, log_variance

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        mean, variance = self.q_mean_variance(x0, t)
        return mean + torch.sqrt(variance) * noise, noise
    
    def p_losses(self, x0, t, condition, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        x_noisy, _ = self.q_sample(x0, t, noise)
        pred_noise = self.denoise_fn(x_noisy, t, condition)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_noise)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(noise, pred_noise)
        else:
            raise NotImplementedError()

        return loss

    def compute_loss(self, x0, condition):
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                x0 = self.vqgan.encode(x0, quantize=False)
                condition = self.vqgan.encode(condition, quantize=False)
                # normalize to -1 and 1
                x0 = ((x0 - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
                condition = ((condition - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            print("Hi")
            x0 = normalize_img(x0)
            condition = normalize_img(condition)

        b = x0.shape[0]
        t = self.sample_timesteps(b).long().to(self.device)
        return self.p_losses(x0, t, condition)

def normalize_img(t):
    return t * 2 - 1
    
def unnormalize_img(t):
    return (t + 1) * 0.5



    # @torch.inference_mode()
    # def p_sample_ddim(self, model, x_t, t_index, t_next, n, condition):
    #     t = (torch.ones(n) * t_index).long().to(self.device)
    #     t_m1 = (torch.ones(n) * t_next).long().to(self.device)
    #     predicted_noise = model(x_t, t, condition)

    #     alpha_bar = self.alpha_bar[t][:, None, None, None]
    #     one_minus_alpha_bar = 1. - alpha_bar
    #     alpha_bar_next = self.alpha_bar[t_m1][:, None, None, None]

    #     x0_pred = 1/torch.sqrt(alpha_bar)*(x_t - torch.sqrt(one_minus_alpha_bar)*predicted_noise)
    #     x0_pred -= torch.mean(x0_pred, dim = (2,3),keepdim = True)
    #     x0_pred /= torch.std(x0_pred, dim = (2,3), correction = 0, keepdim = True)

    #     eta = self.ddim_eta
    #     sigma = eta*((1 - alpha_bar/alpha_bar_next)*(1 - alpha_bar_next)/(1 - alpha_bar)).sqrt()
    #     c = (1 - alpha_bar_next - sigma**2).sqrt()

    #     mean = alpha_bar_next.sqrt()*x0_pred + c*predicted_noise
    #     if t_next < 0:
    #         return x0_pred
    #     else:
    #         noise = torch.randn_like(x_t)
    #         return mean + sigma*noise

    # @torch.inference_mode()
    # def p_sample_loop_ddim(self, model, x_t, t_index, condition):
    #     n = x_t.shape[0]
    #     ddim_sampling_steps = self.sampling_timesteps
    #     logging.info(f"Sampling {n} new images using DDIM with the number of steps tau = {ddim_sampling_steps}....")

    #     times = torch.linspace(-1, t_index-1, steps = ddim_sampling_steps+1) # [-1, 0, 1, 2, ..., T-1] when ddim_sampling_steps == T
    #     logging.info(f"Sampling time steps {times}")
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (2, 1), (1, 0), (0, -1)]

    #     x = x_t.clone()
    #     loop = [x]
    #     for t, t_m1 in time_pairs:
    #         x = self.p_sample_ddim(model, x, t, t_m1, n, condition)
    #         loop.append(x)
    #     x_trajec = torch.cat([loop[i] for i in range(ddim_sampling_steps+1)], dim = 1)
    #     return x_trajec
