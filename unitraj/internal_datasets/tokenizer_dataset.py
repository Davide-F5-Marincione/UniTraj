from .base_dataset import BaseDataset


class TokenizerDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, is_test=False):
        super().__init__(config, is_validation, is_test)
