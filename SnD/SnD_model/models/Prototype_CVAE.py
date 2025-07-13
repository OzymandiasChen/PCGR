from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import copy
from transformers.activations import gelu
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.bert.modeling_bert import *
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D

from SnD_model.models.utilis import AverageSelfAttention, Decoder, MLP_ReLU, MLP_stable






class Prototype_CVAEModel(nn.Module):

    def __init__(self, config, args):
        super(Prototype_CVAEModel, self).__init__()
        self.args = args
        self.latent_dim = self.args.z_dim
        self.mlp_dim = self.args.mlp_bottleneck_dim


        self.prior_mean = MLP_ReLU(config.n_embd, self.latent_dim, self.mlp_dim)
        self.prior_logvar = MLP_ReLU(config.n_embd, self.latent_dim, self.mlp_dim)
        self.post_mean = MLP_ReLU(config.n_embd, self.latent_dim, self.mlp_dim)
        self.post_logvar = MLP_ReLU(config.n_embd, self.latent_dim, self.mlp_dim)
        self.avg_attn = AverageSelfAttention(config.n_embd)
        self.latent_mlp = nn.Linear(self.latent_dim, config.n_embd, bias=False)

    def initialize(self, path):
        self.decoder = Decoder.from_pretrained(path)
        self.encoder = self.decoder.transformer

    
    @staticmethod
    def reparameterize(mu, logvar, nsamples=1):      

        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def kl_loss(self, mean1, logvar1, mean2, logvar2): 
        exponential = logvar1 - logvar2 - \
            torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
            torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, input_ids=None, attention_mask=None, px_tokens=None, px_mask=None, prompt_tokens=None, prompt_mask=None, 
                task_ids=None, mem_std=None, prototype=None, phase='train'):

        if phase=='train': 
            if px_tokens.size() != px_mask.size() or prompt_tokens.size() != prompt_mask.size() or input_ids.size() != attention_mask.size():
                raise ValueError('Sizes of input and its mask not match.')
            
            post_out = self.encoder(input_ids=px_tokens, attention_mask=px_mask)
            post_emb, _ = self.avg_attn(post_out[0], attention_mask=px_mask)
            posterior_mean, posterior_logvar = self.post_mean(post_emb), self.post_logvar(post_emb)

            prototype_proxy, _ = self.get_prototype_proxy( px_tokens, px_mask, task_ids)
            prior_mean, prior_logvar = self.prior_mean(prototype_proxy), self.prior_logvar(prototype_proxy)

            latent_z = self.reparameterize(posterior_mean, posterior_logvar)
            latent_proj = self.args.alpha_z * self.latent_mlp(latent_z)         
            assert not torch.isnan(latent_z).any(), 'Training gets NAN z!'


            kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
            dec_out = self.decoder(input_ids=input_ids, latent_proj=latent_proj)
            outputs = (dec_out.logits, dec_out.past_key_values)

            return outputs + (kl_loss,)    

        if phase=='inference':
            prior_mean, prior_logvar = self.prior_mean(prototype), self.prior_logvar(prototype)

            latent_z = self.reparameterize(prior_mean, prior_logvar)
            latent_proj = self.args.alpha_z * self.latent_mlp(latent_z)         
            if torch.isnan(latent_z).any():
                print('Nan in inference')
                return (None,)

            return (latent_proj)            


    def get_prototype_proxy(self, px_tokens, px_mask, task_ids):
        with torch.no_grad():
            post_out = self.encoder(input_ids=px_tokens, attention_mask=px_mask)
            post_emb, _ = self.avg_attn(post_out[0], attention_mask=px_mask) 
            tasks_unique, _ = torch.sort(torch.unique(task_ids))
            tasks_unique = tasks_unique.tolist()
            task_ebd_dict = {}
            for task_id in tasks_unique:
                task_ebd_dict[task_id] = {}
                mask = (task_ids == task_id).unsqueeze(1).expand_as(post_emb)
                task_ebd_dict[task_id]['ebds'] = torch.masked_select(post_emb, mask).view(-1, post_emb.size(1))
                task_ebd_dict[task_id]['mean'] = torch.mean(task_ebd_dict[task_id]['ebds'], dim=0)
                task_ebd_dict[task_id]['logvar'] = torch.log(torch.var(task_ebd_dict[task_id]['ebds'], dim=0, unbiased=False))
            prototype_proxy = torch.stack([task_ebd_dict[i.item()]['mean'] for i in task_ids], dim=0)
            return prototype_proxy, task_ebd_dict


class PrevPrototype_CVAEModel(Prototype_CVAEModel):
    def __init__(self, config, args):
        super(PrevPrototype_CVAEModel, self).__init__(config, args)


