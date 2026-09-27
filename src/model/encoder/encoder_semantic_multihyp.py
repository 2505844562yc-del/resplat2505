from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from ..semantic_multihyp import (
    BoundaryAwareRayMixtureDecoder,
    CrossViewHypothesisVerifier,
    CrossViewVerifierCfg,
    GaussianAssemblerCfg,
    HypothesisGaussianAssembler,
    RayMixtureCfg,
    SemanticMultiHypothesisInitializer,
    SemanticSceneEncoder,
    SemanticSceneEncoderCfg,
)
from ..semantic_multihyp.types import GeometryEvidence, GeometryProviderOutput
from .encoder_resplat import EncoderReSplat, EncoderReSplatCfg


@dataclass
class EncoderSemanticMultiHypCfg(EncoderReSplatCfg):
    name: Literal["semantic_multihyp"]
    semantic_encoder: SemanticSceneEncoderCfg
    ray_mixture: RayMixtureCfg
    cross_view_verifier: CrossViewVerifierCfg
    gaussian_assembler: GaussianAssemblerCfg


class EncoderSemanticMultiHyp(EncoderReSplat):
    """ReSplat-compatible encoder with a new semantic topology head.

    Base parameter names remain unchanged, allowing direct loading of released
    ReSplat checkpoints. All modules constructed by EncoderReSplat are frozen;
    only semantic/multi-hypothesis modules added afterwards are trainable.
    """

    cfg: EncoderSemanticMultiHypCfg

    def __init__(self, cfg: EncoderSemanticMultiHypCfg) -> None:
        if cfg.num_refine != 0:
            raise ValueError("semantic_multihyp does not use recurrent refinement")
        cfg.return_geometry_evidence = True
        super().__init__(cfg)

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.semantic_initializer = SemanticMultiHypothesisInitializer(
            geometry_provider=None,
            semantic_encoder=SemanticSceneEncoder(cfg.semantic_encoder),
            ray_decoder=BoundaryAwareRayMixtureDecoder(cfg.ray_mixture),
            verifier=CrossViewHypothesisVerifier(cfg.cross_view_verifier),
            assembler=HypothesisGaussianAssembler(cfg.gaussian_assembler),
        )

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for name, child in self.named_children():
            if name == "semantic_initializer":
                child.train(mode)
            else:
                child.eval()
        return self

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        renderer=None,
    ):
        with torch.no_grad():
            base = super().forward(
                context,
                global_step,
                deterministic=deterministic,
                visualization_dump=visualization_dump,
                scene_names=scene_names,
                renderer=renderer,
            )
        if not isinstance(base, dict) or "geometry_evidence" not in base:
            raise RuntimeError("base encoder did not expose geometry evidence")
        raw = base["geometry_evidence"]
        geometry = GeometryProviderOutput(
            gaussians=base["gaussians"],
            depths=base["depths"],
            evidence=GeometryEvidence(
                depth_probabilities=raw["depth_probabilities"],
                depth_candidates=raw["depth_candidates"],
                features=raw["features"],
                semantic_backbone_features=tuple(raw["semantic_backbone_features"]),
            ),
        )
        semantic_output = self.semantic_initializer.forward_from_geometry(
            context,
            geometry,
            hard_gate=False,
        )
        return {
            "gaussians": semantic_output.assembly.gaussians,
            "depths": base["depths"],
            "semantic_multihyp": semantic_output,
        }
