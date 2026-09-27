from typing import Optional

from .encoder import Encoder
from .encoder_resplat import EncoderReSplat, EncoderReSplatCfg
from .encoder_semantic_multihyp import EncoderSemanticMultiHyp, EncoderSemanticMultiHypCfg
from .visualization.encoder_visualizer import EncoderVisualizer
ENCODERS = {
    "resplat": (EncoderReSplat, None),
    "semantic_multihyp": (EncoderSemanticMultiHyp, None),
}

EncoderCfg = EncoderSemanticMultiHypCfg | EncoderReSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
