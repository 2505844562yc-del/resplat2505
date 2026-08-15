from .loss import Loss
from .loss_boundary import LossBoundary, LossBoundaryCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossBoundaryCfgWrapper: LossBoundary,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = (
    LossBoundaryCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
