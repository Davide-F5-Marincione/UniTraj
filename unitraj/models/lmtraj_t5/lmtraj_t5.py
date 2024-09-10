
import nltk
import torch
import logging
import evaluate
import numpy as np
from filelock import FileLock

from transformers.utils import is_offline_mode
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from unitraj.models.base_model.base_model import BaseModel


def init_nltk():
    r"""Initialize nltk data files"""
    
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rouge score expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels



# class AutoBotEgo(nn.Module):
class LMTrajT5(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config):

        super().__init__(config)

        # Initialize the Natural language toolkit
        init_nltk()

        self.config = config

        auto_config = AutoConfig.from_pretrained(config['model_config_name'] if config['model_config_name']  else config['inner_model_name'] , trust_remote_code=False, cache_dir=config['cache_dir'])

        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'] if config['tokenizer_name'] else config['inner_model_name'], trust_remote_code=False, cache_dir=config['cache_dir'], use_fast=not config['use_slow_tokenizer'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config['inner_model_name'], config=auto_config, trust_remote_code=False, cache_dir=config['cache_dir'])

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        if config['tokenizer_name'] is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess_function(self, batch):
        batch_size = batch['batch_size']

        inputs = batch['input_dict']

        obs_traj = inputs['obj_trajs'][..., :2].cpu().numpy()
        track_index_to_predict = inputs['track_index_to_predict'].cpu().numpy()

        center_gt = inputs['center_gt_trajs'][..., :2].cpu().numpy()

        inp = []
        out = []
        for ped_idx in range(batch_size):
            inp.append(' '.join([f"{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 0]:.2f}|{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 1]:.2f}" for i in range(21)]))
            out.append(' '.join([f"{center_gt[ped_idx, i, 0]:.2f}|{center_gt[ped_idx, i, 1]:.2f}" for i in range(60)]))

        model_inputs = self.tokenizer(inp, max_length=self.config['max_source_length'], padding="max_length", truncation=True)
        labels = self.tokenizer(text_target=out, max_length=self.config['max_target_length'], padding="max_length", truncation=True)
        labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs['input_ids'] = torch.as_tensor(model_inputs['input_ids']).to(inputs['obj_trajs'].device)
        model_inputs['labels'] = torch.as_tensor(labels["input_ids"]).to(inputs['obj_trajs'].device)
        model_inputs['attention_mask'] = torch.as_tensor(model_inputs['attention_mask']).to(inputs['obj_trajs'].device)
        return model_inputs

    def forward(self, batch):
        outputs = self.model(**self.preprocess_function(batch))
        return outputs.loss
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['batch_size'], prog_bar=True)
        return loss
    
    def generate(self, batch):
        batch = self.preprocess_function(batch)
        generated_tokens = self.model.generate(**batch, max_length=self.config["max_target_length"], num_beams=1)

        generated_tokens = generated_tokens.cpu().numpy()

        labels = batch['labels'].cpu().numpy()


        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

        if not self.config.use_slow_tokenizer:
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            filtered_tokens_preds = np.where(generated_tokens >= self.tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
            decoded_preds = self.tokenizer.sp_model.decode(filtered_tokens_preds.tolist())
            filtered_tokens_labels = np.where(labels >= self.tokenizer.sp_model.get_piece_size(), 0, labels)
            decoded_labels = self.tokenizer.sp_model.decode(filtered_tokens_labels.tolist())
        
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        metric = evaluate.load("rouge")
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        # To suppress rouge's annoying "Using default tokenizer." line
        logging.disable(logging.CRITICAL)
        result = metric.compute(use_stemmer=True)
        logging.disable(logging.NOTSET)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        return result, decoded_labels
    
    def validation_step(self, batch, batch_idx, status='val'):
        result, decoded_labels = self.generate(batch)

        output = {}
        try:
            traj = [[tuple(map(float, snap.split('|'))) for snap in line.split(' ')][:60] for line in decoded_labels]
            output['predicted_trajectory'] = torch.as_tensor(traj).to(batch['input_dict']['obj_trajs'].device)[:, None]
        except:
            traj = [[(0,0)] * 60 for _ in range(len(decoded_labels))]
            output['predicted_trajectory'] = torch.as_tensor(traj).to(batch['input_dict']['obj_trajs'].device)[:, None]
        
        output['predicted_probability'] = torch.ones((len(decoded_labels), 1)).to(batch['input_dict']['obj_trajs'].device) 

        for k, v in result.items():
            self.log(status + "/"+ k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['batch_size'], prog_bar=True)

        self.log_info(batch, batch_idx, output, status=status)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, status='test')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                                        "weight_decay": self.config.weight_decay,},
                                        {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                                        "weight_decay": 0.0,},]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
        return [optimizer], [scheduler]