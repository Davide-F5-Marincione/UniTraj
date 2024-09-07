from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.lmtraj_zero.lmtraj_zero import LMTrajZero
from unitraj.models.lmtraj_zero.lmtraj_zero_gpt import LMTrajZeroGPT
from unitraj.models.lmtraj_t5.lmtraj_t5 import LMTrajT5
from unitraj.models.tokenizer.tokenizer import Tokenizer


__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'lmtraj_zero': LMTrajZero,
    'lmtraj_zero_gpt': LMTrajZeroGPT,
    'lmtraj_t5': LMTrajT5,
    'tokenizer': Tokenizer
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
