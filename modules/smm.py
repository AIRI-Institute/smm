import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from models.utils import *
from models.utils import PositionEmbed
from .attention import Transformer
from .mixture_decoder import Decoder
from .object_discovery_encoder import Encoder


def norm_prob(mus, logsigmas, values):
    mus = torch.unsqueeze(mus, 2)
    logsigmas = torch.unsqueeze(logsigmas, 2)
    values = torch.unsqueeze(values, 1)
    var = torch.exp(logsigmas) ** 2
    log_prob = (-((values - mus) ** 2) / (2 * var)).sum(dim=-1) - logsigmas.sum(dim=-1) - values.shape[-1] * math.log(
        math.sqrt((2 * math.pi)))
    return torch.exp(log_prob)


class SelfAttentionGMM(nn.Module):
    """
    Slot Attention module
    """

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.mu = nn.Parameter(torch.randn(1, 1, dim))
        self.logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim

        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim * 2)
        self.norm_mu = nn.LayerNorm(dim)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, dim)
        )
        self.heads = 4
        self.transf = Transformer(dim, self.heads, dim // self.heads, depth=2)
        self.mu_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2)
        )
        self.sigma_init = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2)
        )

    def step(self, slots, k, v, b, n, d, device, n_s, pi_cl):
        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)

        # E step
        dots = ((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) ** 2 / torch.unsqueeze(torch.exp(q_logsigma) ** 2,
                                                                                          2)).sum(dim=-1) * self.scale
        dots_exp = (torch.exp(-dots) + self.eps) * pi_cl
        attn = dots_exp / dots_exp.sum(dim=1, keepdim=True)  # gammas
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # M step for mus
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)

        # M step for prior probs of each gaussian
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = pi_cl_new / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)

        # NN update for mus
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = updates_mu.reshape(b, -1, d)
        updates_mu = updates_mu + self.mlp_mu(self.norm_mu(updates_mu))
        if torch.isnan(updates_mu).any():
            print('updates_mu Nan appeared')

        # M step for logsigmas for new mus
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', (
            (torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2)) ** 2 + self.eps, attn)))
        if torch.isnan(updates_logsigma).any():
            print('updates_logsigma Nan appeared')

        # new gaussians params
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)

        log_likelihood = torch.tensor(0, device=slots.device)

        return slots, pi_cl_new, -log_likelihood, attn

    def forward(self, inputs, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        
        n_s = self.num_slots

        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)

        mu = self.mu.expand(b, -1, -1)
        logsigma = self.logsigma.expand(b, -1, -1)

        inputs = torch.cat([mu, logsigma, inputs], dim=1)
        inputs = self.transf(inputs)
        mu, logsigma, inputs = torch.split(inputs, [1, 1, n], dim=1)
        mu = self.mu_init(mu)
        logsigma = self.sigma_init(logsigma)

        mu = mu.expand(-1, n_s, -1)
        sigma = logsigma.exp().expand(-1, n_s, -1)

        slots_init = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        
        slots = slots_init

        for i in range(self.iters):
            slots, pi_cl, log_dict, attn = self.step(slots, k, v, b, n, d, device, n_s, pi_cl)
        slots, pi_cl, log_dict, attn = self.step(slots.detach(), k, v, b, n, d, device, n_s, pi_cl)
        log_l = log_dict
        return self.mlp_out(slots), log_l, attn, self.mlp_out(slots_init)


class FairGMM(nn.Module):
    """
    Slot Attention module
    """

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim * 2))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim * 2))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim

        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim * 2)
        self.norm_mu = nn.LayerNorm(dim)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, dim)
        )

    def step(self, slots, k, v, b, n, d, device, n_s, pi_cl):
        slots_prev = slots

        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)

        # E step
        dots = ((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) ** 2 / torch.unsqueeze(torch.exp(q_logsigma) ** 2,
                                                                                          2)).sum(dim=-1)
        dots_exp = (torch.exp(-dots) + self.eps) * pi_cl
        gammas = dots_exp / dots_exp.sum(dim=1, keepdim=True)  # gammas
        attn = gammas / gammas.sum(dim=-1, keepdim=True)

        # M step for mus
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)

        # NN update for mus
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = updates_mu.reshape(b, -1, d)
        updates_mu = updates_mu + self.mlp_mu(self.norm_mu(updates_mu))
        if torch.isnan(updates_mu).any():
            print('updates_mu Nan appeared')

        # M step for logsigmas for new mus
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', (
            (torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2)) ** 2 + self.eps, attn)))
        if torch.isnan(updates_logsigma).any():
            print('updates_logsigma Nan appeared')

        # new gaussians params
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)

        # M step for prior probs of each gaussian
        pi_cl_new = gammas.sum(dim=-1, keepdim=True)
        pi_cl_new = pi_cl_new / pi_cl_new.shape[2]

        return slots, pi_cl_new

    def forward(self, inputs, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.num_slots

        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_k(inputs)

        for _ in range(self.iters):
            slots, pi_cl = self.step(slots, k, v, b, n, d, device, n_s, pi_cl)
        slots, pi_cl = self.step(slots.detach(), k, v, b, n, d, device, n_s, pi_cl)

        return self.mlp_out(slots)

class GMMEncoder(nn.Module):
    def __init__(
        self, num_iter, num_slots, feature_size,
        slot_size, mlp_size,
        resolution):

        super().__init__()

        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size

        self.pos_emb = PositionEmbed(feature_size, resolution)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, slot_size)
        )

        self.slot_attention = SelfAttentionGMM(num_slots, slot_size, hidden_dim=mlp_size, iters=num_iter)

    def forward(self, f):
        f = self.pos_emb(f)
        f = torch.flatten(f, start_dim=2, end_dim=3).permute(0, 2, 1)
        f = self.mlp(f)

        slots, log_l, attn, slots_init = self.slot_attention(f)
        
        return {
            'slots': slots,
            # 'slots_init': slots_init,
            'attn': attn,
        }


class GMMMixtureDecoderModel(nn.Module):
    def __init__(
            self,
            args,
    ):
        super().__init__()

        self.encoder = Encoder(
            channels=args.encoder_channels,
            strides=args.encoder_strides,
            kernel_size=args.encoder_kernel_size,
        )
        self.decoder = Decoder(
            resolution=args.resolution,
            init_resolution=args.init_resolution,
            slot_size=args.slot_size,
            kernel_size=args.decoder_kernel_size,
            channels=args.decoder_channels,
            strides=args.decoder_strides,
        )

        encoder_stride = 1
        for s in args.encoder_strides:
            encoder_stride *= s

        self.slot_attn = GMMEncoder(args.num_iter, args.num_slots, args.encoder_channels[-1],
                                    args.slot_size, args.mlp_size,
                                    [args.resolution[0] // encoder_stride, args.resolution[1] // encoder_stride],)

        self.num_iter = args.num_iter
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size


    def forward(self, x, sigma=0, is_Train=False):
        B = x.shape[0]
        f = self.encoder(x)
        slot_attn_out = self.slot_attn(f, sigma=sigma)
        slots = slot_attn_out['slots']
        masks, recons = self.decoder(slots)

        recon = torch.sum(recons * masks, dim=1)
        mse = F.mse_loss(recon, x)

        output = {
            "mse": mse,
            "recon": recon,
            "recons": recons,
            "masks": masks,
            # "attns": slot_attn_out['attn'],
            'slots_init': slot_attn_out['slots_init'],
            'slots': slots
        }

        return output

