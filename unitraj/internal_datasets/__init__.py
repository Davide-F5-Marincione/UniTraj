from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .lmtraj_zero_dataset import LMTrajZeroDataset
from .lmtraj_t5_dataset import LMTrajT5Dataset
from .tokenizer_dataset import TokenizerDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    'lmtraj_zero': LMTrajZeroDataset,
    'lmtraj_zero_gpt': LMTrajZeroDataset,
    'lmtraj_t5': LMTrajT5Dataset,
    'tokenizer': TokenizerDataset
}


def build_dataset(config, val=False, test=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val, is_test=test
    )
    return dataset
