import os
import sentencepiece as spm
import json
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn

from unitraj.models.base_model.base_model import BaseModel


# class AutoBotEgo(nn.Module):
class Tokenizer(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.a = nn.Linear(1,1, bias=False)

    def forward(self, batch):
        batch_size = batch['batch_size']
        print("hello!")
 
        inputs = batch['input_dict']
        obs_traj = inputs['obj_trajs'][..., :2].cpu().numpy()
        track_index_to_predict = inputs['track_index_to_predict'].cpu().numpy()
        center_gt = inputs['center_gt_trajs'][..., :2].cpu().numpy()
        prompt = ""
        for ped_idx in (progressbar:=tqdm(range(batch_size), desc='Chatbot started!')):

            inp = ' '.join([f"{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 0]:.2f}|{obs_traj[ped_idx, track_index_to_predict[ped_idx], i, 1]:.2f}" for i in range(21)])
            prompt += inp + '\n'
            out = ' '.join([f"{center_gt[ped_idx, i, 0]:.2f}|{center_gt[ped_idx, i, 1]:.2f}" for i in range(60)])
            prompt += out + '\n'
        
        with open('./tokenizer_input.txt', 'a') as f:
            f.write(prompt)

        llm_processed = torch.zeros((batch_size, 1, 60, 5), dtype=torch.float32, device=batch['input_dict']['obj_trajs'].device) # Pad features
        
        output = {}
        output['predicted_trajectory'] = llm_processed.to(batch['input_dict']['obj_trajs'].device)
        output['predicted_probability'] = torch.ones((batch_size, 1), dtype=torch.float32, device=batch['input_dict']['obj_trajs'].device)

        a = self.a(torch.ones([10, 1], device=batch['input_dict']['obj_trajs'].device))
        return output, (a - a).mean() + 10
        

    def configure_optimizers(self):
        return [torch.optim.SGD(self.a.parameters())], []