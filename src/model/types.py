from dataclasses import dataclass

from jaxtyping import Float, Bool
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"] | None
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"] | None = None
    rotations: Float[Tensor, "batch gaussian 4"] | None = None
    probabilities: Float[Tensor, "batch gaussian distr"] | None = None
    mask: Bool[Tensor, "batch gaussian"] | None = None
    filter_3D: Float[Tensor, "batch gaussian"] | None = None
    rotations_unnorm: Float[Tensor, "batch gaussian 4"] | None = None
    scale_factor: Float[Tensor, "batch"] | None = None
    shift: Float[Tensor, "batch 3"] | None = None
    semantic_features: Float[Tensor, "batch gaussian semantic"] | None = None
    semantic_uncertainty: Float[Tensor, "batch gaussian 1"] | None = None
