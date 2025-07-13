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





class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        if attention_mask is not None:
            scores = scores + attention_mask
        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        if len(weighted.shape) == 2:
            representations = weighted.sum(0).unsqueeze(0)
        else:
            representations = weighted.sum(1).squeeze(1)
        return representations, scores

class Decoder(GPT2LMHeadModel):
    def forward(
        self,
        input_ids=None,
        latent_proj=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.transformer.wte(input_ids)
        if latent_proj is not None:
            inputs_embeds = inputs_embeds + latent_proj         

        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits, 
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )




class MLP_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_hidden_dim):
        super(MLP_ReLU, self).__init__()
        self.m = nn.Sequential(
                            nn.Linear(in_dim, mlp_hidden_dim), 
                            nn.ReLU(), 
                            nn.Linear(mlp_hidden_dim, out_dim)
                        )

    def forward(self, x):
        x = self.m(x)
        return x

