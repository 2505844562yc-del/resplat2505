from dataclasses import dataclass

from torch import Tensor

from ..types import Gaussians


@dataclass
class GeometryEvidence:
    """Frozen geometry signals consumed by the semantic hypothesis modules."""

    depth_probabilities: Tensor  # [B, V, D, H, W]
    depth_candidates: Tensor  # [B, V, D, H, W], metric depth
    features: Tensor  # [B, V, C, H_latent, W_latent]
    semantic_backbone_features: tuple[Tensor, ...]  # each [B*V, C, H/14, W/14]


@dataclass
class GeometryProviderOutput:
    gaussians: Gaussians
    depths: Tensor
    evidence: GeometryEvidence


@dataclass
class RayHypotheses:
    """Two fixed candidate slots; the second slot is softly activated."""

    depths: Tensor  # [B, V, 2, H, W]
    mixture_weights: Tensor  # [B, V, 2, H, W]
    second_active_probability: Tensor  # [B, V, 1, H, W]
    posterior_entropy: Tensor  # [B, V, 1, H, W]
    normalized_separation: Tensor  # [B, V, 1, H, W]
    mode_masses: Tensor  # [B, V, 2, H, W]


@dataclass
class SemanticPredictions:
    """Dense trainable semantics used for geometry decisions, not PCA display."""

    class_logits: Tensor  # [B, V, K, H, W]
    instance_embeddings: Tensor  # [B, V, E, H, W], L2 normalized
    boundary_logits: Tensor  # [B, V, 1, H, W]
    confidence_logits: Tensor  # [B, V, 1, H, W]
    decoder_features: Tensor  # [B, V, C, H/4, W/4]


@dataclass
class CandidateVerification:
    """Cross-view support for each reference-view depth hypothesis."""

    keep_probability: Tensor  # [B, V_ref, K, H, W]
    opacity_delta: Tensor  # [B, V_ref, K, H, W]
    geometric_support: Tensor  # [B, V_ref, K, H, W]
    semantic_support: Tensor  # [B, V_ref, K, H, W]
    visibility_support: Tensor  # [B, V_ref, K, H, W]
    valid_view_count: Tensor  # [B, V_ref, K, H, W]
    view_attention: Tensor  # [B, V_ref, V_target, K, H, W]
