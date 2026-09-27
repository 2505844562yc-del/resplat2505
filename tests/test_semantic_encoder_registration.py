from omegaconf import OmegaConf

from src.config import load_typed_root_config
from src.model.encoder.encoder_semantic_multihyp import EncoderSemanticMultiHypCfg


def test_semantic_encoder_config_composes() -> None:
    # A minimal structural check; the full Hydra composition is exercised by the
    # CLI smoke command because experiment defaults are resolved by Hydra itself.
    cfg = OmegaConf.load("config/model/encoder/semantic_multihyp.yaml")
    assert cfg.name == "semantic_multihyp"
    assert cfg.num_refine == 0
    assert cfg.semantic_encoder.num_classes == 100
