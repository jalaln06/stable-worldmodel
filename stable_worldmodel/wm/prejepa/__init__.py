from .module import CausalPredictor, Embedder  # noqa: F401
from .prejepa import PreJEPA  # noqa: F401


def from_config(
    backbone_name,
    image_size=224,
    patch_size=14,
    is_video_encoder=False,
    history_size=3,
    num_preds=1,
    predictor=None,
    encoding=None,
    extra_dims=None,
    **kwargs,
):
    """Reconstruct a PreJEPA model from a saved training config dict.

    This is the ``_target_`` Hydra uses when a checkpoint was saved by
    ``SaveCkptCallback`` in ``scripts/train/prejepa.py``.
    """
    from collections import OrderedDict

    import stable_pretraining as spt
    from torch import nn
    from transformers import AutoModel, AutoModelForImageClassification

    predictor = predictor or {}
    encoding = encoding or {}
    extra_dims = extra_dims or {}

    # --- Load backbone (mirrors ENCODER_CONFIGS logic in training script) ---
    if backbone_name.startswith("microsoft/resnet-"):
        backbone = AutoModelForImageClassification.from_pretrained(backbone_name)
        backbone.classifier[1] = nn.LayerNorm(backbone.config.hidden_sizes[-1])
        embed_dim = backbone.config.hidden_sizes[-1]
        num_patches = 1
        interp_pos_enc = False
    else:
        backbone = AutoModel.from_pretrained(backbone_name)
        if hasattr(backbone, "vision_model"):  # CLIP-style
            backbone = backbone.vision_model
        embed_dim = backbone.config.hidden_size
        num_patches = (image_size // patch_size) ** 2
        interp_pos_enc = True

    embed_dim += sum(encoding.values())

    # --- Build predictor ---
    predictor_cfg = {k: v for k, v in predictor.items() if k != "size"}
    causal_predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=history_size,
        dim=embed_dim,
        **predictor_cfg,
    )

    # --- Build extra encoders ---
    extra_encoder_modules = nn.ModuleDict(
        OrderedDict(
            (key, Embedder(in_chans=extra_dims[key], emb_dim=emb_dim))
            for key, emb_dim in encoding.items()
        )
    )

    return PreJEPA(
        encoder=spt.backbone.EvalOnly(backbone),
        predictor=causal_predictor,
        extra_encoders=extra_encoder_modules,
        history_size=history_size,
        num_pred=num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )


__all__ = ["CausalPredictor", "Embedder", "PreJEPA", "from_config"]
