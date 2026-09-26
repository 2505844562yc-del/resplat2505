import torch

from src.model.semantic_multihyp import SemanticSceneEncoder, SemanticSceneEncoderCfg


def test_semantic_encoder_outputs_dense_trainable_predictions() -> None:
    cfg = SemanticSceneEncoderCfg(
        backbone_channels=32,
        decoder_channels=24,
        edge_channels=16,
        num_classes=12,
        instance_dim=8,
    )
    encoder = SemanticSceneEncoder(cfg)
    images = torch.rand(2, 3, 3, 64, 96)
    backbone = tuple(
        torch.rand(6, 32, 5, 7)
        for _ in range(4)
    )

    output = encoder(images, backbone)
    assert output.class_logits.shape == (2, 3, 12, 64, 96)
    assert output.instance_embeddings.shape == (2, 3, 8, 64, 96)
    assert output.boundary_logits.shape == (2, 3, 1, 64, 96)
    assert output.confidence_logits.shape == (2, 3, 1, 64, 96)
    assert output.decoder_features.shape[-2:] == (16, 24)

    norms = output.instance_embeddings.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    loss = (
        output.class_logits.mean()
        + output.instance_embeddings.mean()
        + output.boundary_logits.mean()
        + output.confidence_logits.mean()
    )
    loss.backward()
    assert encoder.class_head.weight.grad is not None
    assert encoder.boundary_head.weight.grad is not None
