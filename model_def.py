
# Single entry point for building any model (vqvae: hierarchical or flat).
# It forwards the full config to the unified factory in vqvae.utils.

from configs.config import Config
from vqvae.utils import initialize_model

def build_model(cfg: Config):
    params = {
        "model_type":         cfg.model_type,
        "latent_channel":     cfg.latent_channel,
        "embed_dim":          cfg.embed_dim,# may be None for some models;
        "codebook_dim":       cfg.codebook_dim,
        "codebook_size":      cfg.codebook_size,
        "decay":              cfg.decay,
        "rotation_trick":     cfg.rotation_trick,
        "kmeans_init":        cfg.kmeans_init,
        "learnable_codebook": cfg.learnable_codebook,
        "ema_update":         cfg.ema_update,
        "threshold_dead":     cfg.threshold_dead,
    }
    _warn_if_levels_mismatch(cfg.model_type, cfg.num_levels)
    model = initialize_model(params)
    
    return model

def _warn_if_levels_mismatch(model_type: str, num_levels: int):
    try:
        if "HVQ" in model_type and num_levels < 2:
            print(f"[model_def] warning: model_type={model_type} looks hierarchical but NUM_LEVELS={num_levels}. "
                  f"Did you mean NUM_LEVELS=2?")
        if "Flat" in model_type and num_levels != 1:
            print(f"[model_def] warning: model_type={model_type} looks flat but NUM_LEVELS={num_levels}. "
                  f"Did you mean NUM_LEVELS=1?")
    except Exception:
        pass
