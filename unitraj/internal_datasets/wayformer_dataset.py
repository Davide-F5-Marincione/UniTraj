from .base_dataset import BaseDataset


class WayformerDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, is_test=False):
        super().__init__(config, is_validation, is_test)
