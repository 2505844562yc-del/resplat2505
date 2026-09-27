from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import SemanticMultiHypOutput


@dataclass
class SemanticSupervisionCfg:
    """Weights and label conventions for semantic topology supervision."""

    semantic_weight: float = 1.0
    boundary_weight: float = 0.5
    instance_weight: float = 0.25
    confidence_weight: float = 0.1
    hypothesis_weight: float = 0.5
    second_gate_weight: float = 0.25
    verifier_weight: float = 0.25
    semantic_ignore_label: int = 255
    instance_ignore_label: int = 0
    depth_sigma_relative: float = 0.03
    keep_threshold_relative: float = 0.05
    instance_temperature: float = 0.1
    max_boundary_positive_weight: float = 20.0
    eps: float = 1e-6


class SemanticSupervision(nn.Module):
    """Supervise semantics and both geometry innovations on context views.

    Expected optional context keys are ``semantic`` [B,V,H,W], ``instance``
    [B,V,H,W], and ``depth`` [B,V,H,W]. Missing modalities are skipped, which
    keeps RGB-only datasets usable for infrastructure tests and mixed training.
    Returned losses are already multiplied by their configured weights.
    """

    def __init__(self, cfg: SemanticSupervisionCfg) -> None:
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _resize_labels(labels: Tensor, size: tuple[int, int]) -> Tensor:
        if labels.shape[-2:] == size:
            return labels
        shape = labels.shape
        resized = F.interpolate(
            labels.reshape(-1, 1, *shape[-2:]).float(),
            size=size,
            mode="nearest",
        )
        return resized.reshape(*shape[:-2], *size).to(labels.dtype)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor, eps: float) -> Tensor:
        weights = mask.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(eps)

    def _semantic_valid(self, labels: Tensor, num_classes: int) -> Tensor:
        return (
            (labels >= 0)
            & (labels < num_classes)
            & (labels != self.cfg.semantic_ignore_label)
        )

    def _boundary_target(
        self,
        semantic: Tensor | None,
        instance: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        source = instance if instance is not None else semantic
        if source is None:
            raise ValueError("semantic or instance labels are required")
        if instance is not None:
            valid = instance != self.cfg.instance_ignore_label
        else:
            valid = self._semantic_valid(semantic, 1 << 30)

        boundary = torch.zeros_like(source, dtype=torch.bool)
        pair = valid[..., :, 1:] & valid[..., :, :-1]
        different = (source[..., :, 1:] != source[..., :, :-1]) & pair
        boundary[..., :, 1:] |= different
        boundary[..., :, :-1] |= different
        pair = valid[..., 1:, :] & valid[..., :-1, :]
        different = (source[..., 1:, :] != source[..., :-1, :]) & pair
        boundary[..., 1:, :] |= different
        boundary[..., :-1, :] |= different
        return boundary.unsqueeze(2).float(), valid.unsqueeze(2)

    def _balanced_boundary_loss(
        self,
        logits: Tensor,
        target: Tensor,
        valid: Tensor,
    ) -> Tensor:
        positives = (target * valid).sum()
        negatives = ((1.0 - target) * valid).sum()
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(
            1.0, self.cfg.max_boundary_positive_weight
        )
        weights = torch.where(target > 0.5, positive_weight, 1.0)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return self._masked_mean(loss * weights, valid, self.cfg.eps)

    def _instance_pair_loss(
        self,
        embeddings: Tensor,
        instances: Tensor,
    ) -> Tensor:
        losses = []
        for dim in (-1, -2):
            if dim == -1:
                first, second = embeddings[..., :-1], embeddings[..., 1:]
                id_first, id_second = instances[..., :-1], instances[..., 1:]
            else:
                first, second = embeddings[..., :-1, :], embeddings[..., 1:, :]
                id_first, id_second = instances[..., :-1, :], instances[..., 1:, :]
            similarity = (first * second).sum(dim=2)
            valid = (
                (id_first != self.cfg.instance_ignore_label)
                & (id_second != self.cfg.instance_ignore_label)
            )
            same = (id_first == id_second).to(similarity.dtype)
            logits = similarity / self.cfg.instance_temperature
            pair_loss = F.binary_cross_entropy_with_logits(
                logits, same, reduction="none"
            )
            losses.append(self._masked_mean(pair_loss, valid, self.cfg.eps))
        return torch.stack(losses).mean()

    def forward(
        self,
        output: SemanticMultiHypOutput,
        context: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        semantic = context.get("semantic")
        instance = context.get("instance")
        depth = context.get("depth")
        if semantic is None and instance is None and depth is None:
            return {}

        predictions = output.semantics
        full_size = predictions.class_logits.shape[-2:]
        semantic = (
            self._resize_labels(semantic.long(), full_size)
            if semantic is not None
            else None
        )
        instance = (
            self._resize_labels(instance.long(), full_size)
            if instance is not None
            else None
        )
        losses: dict[str, Tensor] = {}
        boundary_target = boundary_valid = None

        if semantic is not None:
            logits = predictions.class_logits
            valid = self._semantic_valid(semantic, logits.shape[2])
            safe_target = semantic.masked_fill(~valid, 0)
            semantic_loss = F.cross_entropy(
                logits.flatten(0, 1),
                safe_target.flatten(0, 1),
                reduction="none",
            ).reshape_as(semantic)
            losses["class"] = self.cfg.semantic_weight * self._masked_mean(
                semantic_loss, valid, self.cfg.eps
            )
            correct = (logits.argmax(dim=2) == semantic).float().detach()
            confidence_loss = F.binary_cross_entropy_with_logits(
                predictions.confidence_logits.squeeze(2),
                correct,
                reduction="none",
            )
            losses["confidence"] = self.cfg.confidence_weight * self._masked_mean(
                confidence_loss, valid, self.cfg.eps
            )

        if semantic is not None or instance is not None:
            boundary_target, boundary_valid = self._boundary_target(
                semantic, instance
            )
            losses["boundary"] = self.cfg.boundary_weight * self._balanced_boundary_loss(
                predictions.boundary_logits,
                boundary_target,
                boundary_valid,
            )

        if instance is not None:
            losses["instance"] = self.cfg.instance_weight * self._instance_pair_loss(
                predictions.instance_embeddings,
                instance,
            )

        hypothesis_size = output.hypotheses.depths.shape[-2:]
        if boundary_target is not None:
            gate_target = self._resize_labels(boundary_target, hypothesis_size)
            gate_valid = self._resize_labels(boundary_valid, hypothesis_size).bool()
            gate_loss = F.binary_cross_entropy(
                output.hypotheses.second_active_probability.clamp(
                    self.cfg.eps, 1.0 - self.cfg.eps
                ),
                gate_target,
                reduction="none",
            )
            losses["second_gate"] = self.cfg.second_gate_weight * self._masked_mean(
                gate_loss, gate_valid, self.cfg.eps
            )

        if depth is not None:
            depth = self._resize_labels(depth.float(), hypothesis_size)
            valid_depth = torch.isfinite(depth) & (depth > self.cfg.eps)
            candidate_depths = output.hypotheses.depths
            relative_error = (
                (candidate_depths - depth.unsqueeze(2)).abs()
                / depth.unsqueeze(2).clamp_min(self.cfg.eps)
            )
            component_likelihood = torch.exp(
                -0.5 * (relative_error / self.cfg.depth_sigma_relative) ** 2
            )
            likelihood = (
                output.hypotheses.mixture_weights * component_likelihood
            ).sum(dim=2)
            hypothesis_nll = -likelihood.clamp_min(self.cfg.eps).log()
            losses["hypothesis"] = self.cfg.hypothesis_weight * self._masked_mean(
                hypothesis_nll, valid_depth, self.cfg.eps
            )

            keep_target = (
                relative_error < self.cfg.keep_threshold_relative
            ).to(candidate_depths.dtype)
            keep_loss = F.binary_cross_entropy(
                output.verification.keep_probability.clamp(
                    self.cfg.eps, 1.0 - self.cfg.eps
                ),
                keep_target,
                reduction="none",
            )
            keep_valid = valid_depth.unsqueeze(2).expand_as(keep_loss)
            losses["verification"] = self.cfg.verifier_weight * self._masked_mean(
                keep_loss, keep_valid, self.cfg.eps
            )

        return losses
