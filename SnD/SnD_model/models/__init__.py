
from SnD_model.models.Prototype_CVAE import Prototype_CVAEModel, PrevPrototype_CVAEModel




def get_model(config, args):
    VAE = None
    if args.vae_type == 'prototype_cvae':
        VAE = Prototype_CVAEModel(config, args)      

    return VAE


def get_previous_model(config, args):
    prev_VAE = None
    if args.vae_type == 'prototype_cvae':
        prev_VAE = PrevPrototype_CVAEModel(config, args)  

    return prev_VAE
