import numpy as np
import torch

from unitraj.models.base_model.base_model import BaseModel
import transformers
import os
from tqdm import tqdm
import re

max_tries = 20
temperature = 0.7

# class AutoBotEgo(nn.Module):
class LMTrajZero(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.c = config['num_modes']
        self.model_id = config['model_id']
        self.T = config['future_len']
        self.past = config['past_len']
        self.max_new_tokens = config['max_new_tokens']

        os.environ["TOKENIZERS_PARALLELISM"] = 'false'

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id = 128001 # To suppress the warning
        )

        self.prompt_system = config['sys_prompt']
        self.prompt = config['prompt']
        self.coord_template = "({0:.2f}, {1:.2f})"
        self.concurrent_sessions = config['concurrent_sessions']

    def forward(self, batch):
        batch_size = batch['batch_size']

        inputs = batch['input_dict']
        obs_traj = inputs['obj_trajs'][..., :2].cpu().numpy()
        obs_traj_mask = inputs['obj_trajs_mask'].cpu().numpy()
        track_index_to_predict = inputs['track_index_to_predict'].cpu().numpy()

        llm_response_list = ["" for _ in range(batch_size)]
        llm_processed_list = [torch.zeros((self.c, self.T, 2), dtype=torch.float32) for _ in range(batch_size)]

        prepping = []


        for ped_idx in range(batch_size):
            to_track = track_index_to_predict[ped_idx] + 1

            prompt = ''

            for obj_idx in range(obs_traj.shape[1]):
                if obs_traj_mask[ped_idx, obj_idx, 0]:
                    coord_str = '[' + ', '.join([self.coord_template.format(*obs_traj[ped_idx, obj_idx, i]) for i in range(self.past)]) + ']'
                    prompt += self.prompt.format(obj_idx + 1, coord_str) + '\n'
            prompt += f'Forecast: {to_track:02d}'
            prepping.append([{"role": "system", "content": self.prompt_system.format(self.past, self.T, self.c)}, {"role": "user", "content": prompt}])

        todo = set(range(batch_size))
        progress = tqdm(total=batch_size, desc='Running Chatbot', position=0, leave=True)
        while todo:
            current = []
            messages = []
            for _ in range(min(self.concurrent_sessions, len(todo))):
                ped_idx = todo.pop()
                messages.append(prepping[ped_idx])
                current.append(ped_idx)
            
            error_code = ''

            # while tries < max_tries:                
            #     # Set additional information and settings when it kept failing
            #     tmp = 1.0 if tries >= max_tries // 2 else temperature
            #     if tries == max_tries // 4 and add_info < 1:
            #         messages[-1]['content'] += f'\nProvide {self.c:d} hypothetical scenarios based on different extrapolation methods.'
            #         add_info = 1
            #     elif tries == (max_tries // 4) * 2 and add_info < 2:
            #         messages[-1]['content'] += '\nYou can use methods like linear interpolation, polynomial fitting, moving average, and more.'
            #         add_info = 2
                
            # Run the Chatbot model
            try:
                responses = self.pipeline(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                )
                responses = [response[0]["generated_text"][-1]['content'] for response in responses]
                # print('Coord_string--------------------\n', coord_str)
                # print('Response------------------------\n', response)
            except Exception as err:
                error_code = f"Unexpected {err=}, {type(err)=}"
                print(error_code)
                responses = ['' for _ in range(len(messages))]
                todo.update(current)
                continue

            for i, response in enumerate(responses):
                ped_idx = current[i]
                to_track = track_index_to_predict[ped_idx] + 1

                # filter out wrong answers
                if (not (abs(obs_traj[ped_idx, to_track, 0] - obs_traj[ped_idx, to_track, -1]).sum() < 0.3
                    or abs(obs_traj[ped_idx, to_track, 0] - obs_traj[ped_idx, to_track, 2]).sum() < 0.2
                    or abs(obs_traj[ped_idx, to_track, -3] - obs_traj[ped_idx, to_track, -1]).sum() < 0.2) and '[' + self.coord_template.format(*obs_traj[ped_idx, to_track, 0]) in response):
                    error_code = 'Obs coordinates included'
                    todo.add(ped_idx)
                    continue
                elif ('[' + self.coord_template.format(*obs_traj[ped_idx, to_track, 0, ::-1]) in response
                    or '(x4, y4)]' in response
                    or ')]' not in response):
                    error_code = 'Invalid response shape'
                    todo.add(ped_idx)
                    continue
                elif len(response) == 0:
                    error_code = 'Empty response'
                    todo.add(ped_idx)
                    continue
                
                # Convert to list, check validity
                try:
                    response_cleanup = re.sub('[^0-9()\[\],.\-\n]', '', response.replace(':', '\n')).replace('(', '[').replace(')', ']')
                    response_cleanup = [eval(line) for line in response_cleanup.split('\n') if len(line) > 20 and line.startswith('[[') and line.endswith(']]')]
                except:
                    error_code = 'Response to list failed'
                    todo.add(ped_idx)
                    continue
                
                # Remove repeated obs sequence or truncate the response
                if len(response_cleanup) >= self.c:
                    response_cleanup = response_cleanup[-self.c:]
                
                # Check validity
                if (len(response_cleanup) == self.c
                    and all(len(response_cleanup[i]) == self.T for i in range(self.c))
                    and all(all(len(response_cleanup[i][j]) == 2 for j in range(self.T)) for i in range(self.c))):
                    # Add the response to the dump list
                    response_cleanup = torch.as_tensor(response_cleanup, dtype=torch.float32)
                    llm_processed_list[ped_idx] = response_cleanup
                    llm_response_list[ped_idx] += response
                else:
                    error_code = 'Wrong response format'
                    todo.add(ped_idx)
                    continue
            
            progress.update(batch_size - len(todo))

        llm_processed = torch.stack(llm_processed_list)
        llm_processed = torch.cat([llm_processed, torch.zeros((batch_size, self.c, self.T, 3), dtype=torch.float32, device=llm_processed.device)], dim=-1) # Pad features
        
        output = {}
        output['predicted_trajectory'] = llm_processed.to(batch['input_dict']['obj_trajs'].device)
        output['predicted_probability'] = torch.ones((batch_size, self.c), dtype=torch.float32, device=batch['input_dict']['obj_trajs'].device) / self.c

        return output, 0

    def configure_optimizers(self):
        return [], []