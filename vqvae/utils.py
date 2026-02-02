# vqvae/utils.py


import os
import random
import numpy as np
import torch

import vqvae.hierarchical.h_models as hierarchical_models
# import vqvae.hierarchical.vanilla_hmodels as vanilla_hmodels
import vqvae.flat.flat_models as flat_models
# import vqvae.flat.vanilla_fmodels as vanilla_fmodels



def initialize_model(params):
    """
    Dynamically locate the class named by params['model_type'] in any of the
    known modules (hierarchical / flat, vanilla or enhanced), and instantiate it.
    """
    model_type = params["model_type"]
    for module in (hierarchical_models, flat_models):
        if hasattr(module, model_type):
            cls = getattr(module, model_type)
            return cls(
                in_channel=3,
                channel=params["latent_channel"],
                n_res_block=2,
                n_res_channel=params["latent_channel"] // 2,
                embed_dim=params.get("embed_dim", None),
                codebook_dim=params["codebook_dim"],
                n_embed=params["codebook_size"],
                decay=params["decay"],
                rotation_trick=params["rotation_trick"],
                kmeans_init=params["kmeans_init"],
                learnable_codebook=params["learnable_codebook"],
                ema_update=params["ema_update"],
                threshold_ema_dead_code=params.get("threshold_dead", None),
            )
    raise ValueError(f"Model type '{model_type}' not found in available modules.")

def init_seeds(seed=None):
    seed = random.randint(0, 2147483647) if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
