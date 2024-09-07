import numpy as np
import torch

import transformers
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler
from unitraj.models.base_model.base_model import BaseModel
import transformers
import os
from tqdm import tqdm
import re
import nltk
from filelock import FileLock
from transformers.utils import is_offline_mode
import evaluate


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
        for ped_idx in (progressbar:=tqdm(range(batch_size), desc='Chatbot started!')):

            inp.append(' '.join([f"{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 0]:.2f}|{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 1]:.2f}" for i in range(21)]))
            out.append(' '.join([f"{center_gt[ped_idx, i, 0]:.2f}|{center_gt[ped_idx, i, 1]:.2f}" for i in range(60)]))

        padding = "max_length"
        model_inputs = self.tokenizer(inp, max_length=self.config['max_source_length'], padding=padding, truncation=True)
        labels = self.tokenizer(text_target=out, max_length=self.config['max_target_length'], padding=padding, truncation=True)
        
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['input_ids'] = torch.as_tensor(model_inputs['input_ids']).to(inputs['obj_trajs'].device)
        model_inputs['labels'] = torch.as_tensor(model_inputs['labels']).to(inputs['obj_trajs'].device)
        model_inputs['attention_mask'] = torch.as_tensor(model_inputs['attention_mask']).to(inputs['obj_trajs'].device)
        return model_inputs

    def forward(self, batch):
        outputs = self.model(**self.preprocess_function(batch))
        return outputs.loss
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['batch_size'])
        return loss
    
    def generate(self, batch):
        batch = self.preprocess_function(batch)
        generated_tokens = self.model.generate(**batch)

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

        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        return result
    
    def validation_step(self, batch, batch_idx):
        result = self.generate(batch)

        for k, v in result.items():
            self.log("val/"+ k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['batch_size'])

    
    def log_info(self, batch, batch_idx, prediction, status='train'):
        ## logging
        # Split based on dataset
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']
        
        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()

        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)
        # Calculate FDE losses
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)

        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob)

        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            'brier_fde': brier_fde}

        important_metrics = list(loss_dict.keys())

        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in loss_dict.keys():
                new_dict[dataset_name + '/' + key] = loss_dict[key][batch_idx_for_this_dataset]

        # merge new_dict with log_dict
        loss_dict.update(new_dict)
        # loss_dict.update(avg_dict)

        if status == 'val' and self.config.get('eval', False):

            # Split scores based on trajectory type
            new_dict = {}
            trajectory_types = inputs["trajectory_type"].cpu().numpy()
            trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                                         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                                         6: "left_u_turn", 7: "left_turn"}
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types == traj_type)[0]
                if len(batch_idx_for_traj_type) > 0:
                    for key in important_metrics:
                        new_dict["traj_type/" + trajectory_correspondance[traj_type] + "_" + key] = loss_dict[key][
                            batch_idx_for_traj_type]
            loss_dict.update(new_dict)

            # Split scores based on kalman_difficulty @6s
            new_dict = {}
            kalman_difficulties = inputs["kalman_difficulty"][:,
                                  -1].cpu().numpy()  # Last is difficulty at 6s (others are 2s and 4s)
            for kalman_bucket, (low, high) in {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}.items():
                batch_idx_for_kalman_diff = \
                    np.where(np.logical_and(low <= kalman_difficulties, kalman_difficulties < high))[0]
                if len(batch_idx_for_kalman_diff) > 0:
                    for key in important_metrics:
                        new_dict["kalman/" + kalman_bucket + "_" + key] = loss_dict[key][batch_idx_for_kalman_diff]
            loss_dict.update(new_dict)

            new_dict = {}
            agent_types = [1, 2, 3]
            agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}
            for type in agent_types:
                batch_idx_for_type = np.where(inputs['center_objects_type'] == type)[0]
                if len(batch_idx_for_type) > 0:
                    for key in important_metrics:
                        new_dict["agent_types" + '/' + agent_type_dict[type] + "_" + key] = loss_dict[key][
                            batch_idx_for_type]
            # merge new_dict with log_dict
            loss_dict.update(new_dict)

        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}

        for k, v in loss_dict.items():
            self.log(status + "/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])

        if status == 'val' and batch_idx == 0 and not self.config.debug:
            img = visualization.visualize_prediction(batch, prediction)
            wandb.log({"prediction": [wandb.Image(img)]})

        return

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