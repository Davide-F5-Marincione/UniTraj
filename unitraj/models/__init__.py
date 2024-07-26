from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.lmtraj_zero.lmtraj_zero import LMTrajZero

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'lmtraj_zero': LMTrajZero
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
