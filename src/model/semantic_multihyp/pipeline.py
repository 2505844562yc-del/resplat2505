import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cross_view_verifier import CrossViewHypothesisVerifier
from .gaussian_assembler import HypothesisGaussianAssembler
from .ray_mixture import BoundaryAwareRayMixtureDecoder
from .semantic_encoder import SemanticSceneEncoder
from .types import GeometryProviderOutput, SemanticMultiHypOutput


class SemanticMultiHypothesisInitializer(nn.Module):
    """End-to-end initialization path without recurrent refinement."""

    def __init__(
        self,
        geometry_provider: nn.Module,
        semantic_encoder: SemanticSceneEncoder,
        ray_decoder: BoundaryAwareRayMixtureDecoder,
        verifier: CrossViewHypothesisVerifier,
        assembler: HypothesisGaussianAssembler,
    ) -> None:
        super().__init__()
        self.geometry_provider = geometry_provider
        self.semantic_encoder = semantic_encoder
        self.ray_decoder = ray_decoder
        self.verifier = verifier
        self.assembler = assembler

    @staticmethod
    def _resize_depth_distribution(
        probabilities: Tensor,
        candidates: Tensor,
        size: tuple[int, int],
        eps: float = 1e-8,
    ) -> tuple[Tensor, Tensor]:
        if probabilities.shape != candidates.shape:
            raise ValueError("depth posterior and candidate grids must match")
        b, v, d, h, w = probabilities.shape
        if (h, w) == size:
            normalized = probabilities / probabilities.sum(dim=2, keepdim=True).clamp_min(eps)
            return normalized, candidates
        probabilities = F.interpolate(
            probabilities.reshape(b * v, d, h, w).float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, d, *size)
        candidates = F.interpolate(
            candidates.reshape(b * v, d, h, w).float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, d, *size)
        probabilities = probabilities.clamp_min(0)
        probabilities = probabilities / probabilities.sum(dim=2, keepdim=True).clamp_min(eps)
        return probabilities, candidates

    def forward(
        self,
        context: dict[str, Tensor],
        global_step: int = 0,
        deterministic: bool = False,
        hard_gate: bool = False,
    ) -> SemanticMultiHypOutput:
        geometry: GeometryProviderOutput = self.geometry_provider(
            context,
            global_step=global_step,
            deterministic=deterministic,
        )
        semantics = self.semantic_encoder(
            context["image"],
            geometry.evidence.semantic_backbone_features,
        )
        latent_size = geometry.evidence.features.shape[-2:]
        probabilities, candidates = self._resize_depth_distribution(
            geometry.evidence.depth_probabilities,
            geometry.evidence.depth_candidates,
            latent_size,
        )
        hypotheses = self.ray_decoder(
            probabilities,
            candidates,
            semantics.boundary_logits.sigmoid(),
            semantics.confidence_logits.sigmoid(),
        )
        verification = self.verifier(
            hypotheses,
            probabilities,
            candidates,
            semantics,
            context["intrinsics"],
            context["extrinsics"],
        )
        assembly = self.assembler(
            geometry.gaussians,
            hypotheses,
            verification,
            context["intrinsics"],
            context["extrinsics"],
            hard_gate=hard_gate,
        )
        return SemanticMultiHypOutput(
            assembly=assembly,
            semantics=semantics,
            hypotheses=hypotheses,
            verification=verification,
            base_depths=geometry.depths,
            geometry_evidence=geometry.evidence,
        )
