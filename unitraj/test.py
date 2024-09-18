import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from models import build_model
from internal_datasets import build_dataset
from utils.utils import set_seed
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
   
    model = build_model(cfg)

    test_set = build_dataset(cfg, test=True)

    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // test_set.data_chunk_size, 1)

    call_backs = []

    set_loader = DataLoader(
        test_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=test_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=CSVLogger("unitraj", name=cfg.exp_name),
        devices=1,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs,
        num_nodes=1,
    )

    if cfg.ckpt_path is None and not cfg.debug:
        cfg.ckpt_path = find_latest_checkpoint(os.path.join('unitraj', cfg.exp_name, 'checkpoints'))

    trainer.test(model=model, dataloaders=set_loader)


if __name__ == '__main__':
    train()
