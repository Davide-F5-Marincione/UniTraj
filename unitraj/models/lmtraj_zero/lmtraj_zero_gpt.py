import numpy as np
import torch

from unitraj.models.base_model.base_model import BaseModel
import transformers
import os
from tqdm import tqdm
import re
import ast
import openai

max_tries = 20
temperature = 0.7
free_trial = False

# chatbot config
max_timeout = 20

# API config
OPENAI_API_KEY = ['YOU WISH'][0]
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
)

# class AutoBotEgo(nn.Module):
class LMTrajZeroGPT(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.c = 1 #20 #config['num_modes']
        self.model_id = config['model_id']
        self.T = config['future_len']
        self.past = config['past_len']
        self.max_new_tokens = config['max_new_tokens']


        #self.prompt_system = "You are a helpful assistant that Extrapolate the coordinate sequence data. " #config['sys_prompt']
        self.prompt_system = "You are a helpful assistant that extrapolates the coordinate sequence data. The last coordinate is centered at (0,0) and should continue from there. Never output the same coordinate twice."

        #self.prompt = ["Forecast the next {1:d} (x, y) coordinates using the observed {0:d} (x, y) coordinate list.\nDirectly output the 5 different cases of the future {1:d} coordinate Python list without explanation.\nList must be in single line, without line split or numbering.\nDirectly output final lists with 60 x-y coordinates each without for loop or multiply operators.\n{2:s}",
        #            ] #config['prompt']
        self.prompt = ["Forecast the next {1:d} (x, y) coordinates using the observed {0:d} (x, y) coordinate list.\nDirectly output the future {1:d} coordinate Python list without explanation.\nList must be in single line, without line split or numbering.\nDirectly output final lists with 60 x-y coordinates each without for loop or multiply operators.\n{2:s}",
                   ] #config['prompt']
        self.coord_template = "({0:.2f}, {1:.2f})" #!"{2:d}: ({0:.2f}, {1:.2f})"

    def forward(self, batch):
        batch_size = batch['batch_size']

        inputs = batch['input_dict']
        obs_traj = inputs['obj_trajs'][..., :2].cpu().numpy()
        obs_traj_mask = inputs['obj_trajs_mask'].cpu().numpy()
        track_index_to_predict = inputs['track_index_to_predict'].cpu().numpy()
        

        llm_processed_list = torch.stack([torch.zeros((self.c,  self.T, 2), dtype=torch.float32) for _ in range(batch_size)])
        llm_response_list = ['' for _ in range(obs_traj.shape[1])]*batch_size


        for ped_idx in (progressbar:=tqdm(range(batch_size), desc='Chatbot started!')):
            #ped_idx=52
            to_track = track_index_to_predict[ped_idx] + 1

            messages = [{"role": "system", "content": self.prompt_system.format(self.T, self.past)}]
            prompt = []
            
            #for obj_idx in range(obs_traj.shape[1]):
            #    if obj_idx + 1 == to_track:
            #        if obs_traj_mask[ped_idx, obj_idx, 0]:
            #            coord_str = '[' + ', '.join([self.coord_template.format(*obs_traj[ped_idx, to_track, i]) for i in range(self.past)]) + ']'
            #            prompt = self.prompt[0].format(self.past, self.T, coord_str)
            #        else:
            #            continue
            coord_str = '[' + ', '.join([self.coord_template.format(*obs_traj[ped_idx, 0, i]) for i in range(self.past)]) + ']'
            
            prompt = self.prompt[0].format(self.past, self.T, coord_str)

            messages.append({"role": "user", "content": prompt})
            
            error_code = ''
            tries = 0
            add_info = 0

            while tries < max_tries:
                # Prevent RateLimitError by waiting for 20 seconds
                progressbar.set_description('Scene {}/{}, retry {}/{} {}'.format(ped_idx+1, batch_size, tries, max_tries, error_code))
                
                # Set additional information and settings when it kept failing
                tmp = 1.0 if tries >= max_tries // 2 else temperature
                if tries == max_tries // 4 and add_info < 1:
                    messages[-1]['content'] += f'\nProvide {self.c:d} hypothetical scenarios based on different extrapolation methods.'
                    add_info = 1
                elif tries == (max_tries // 4) * 2 and add_info < 2:
                    messages[-1]['content'] += '\nYou can use methods like linear interpolation, polynomial fitting, moving average, and more.'
                    add_info = 2
                
                # Run the Chatbot model
                try:
                    response =  client.chat.completions.create(model=self.model_id, messages=messages, temperature=tmp)
                    response = response.choices[0].message.content.strip()

                    
                except Exception as err:
                    error_code = f"Unexpected {err=}, {type(err)=}"
                    print(error_code)
                    response = ''
                    tries += 1
                    continue

                # filter out wrong answers
                if (not (abs(obs_traj[ped_idx, to_track, 0] - obs_traj[ped_idx, to_track, -1]).sum() < 0.3
                    or abs(obs_traj[ped_idx, to_track, 0] - obs_traj[ped_idx, to_track, 2]).sum() < 0.2
                    or abs(obs_traj[ped_idx, to_track, -3] - obs_traj[ped_idx, to_track, -1]).sum() < 0.2) and #0.2
                    '[' + self.coord_template.format(*obs_traj[ped_idx, to_track, 0], -self.past) in response):
                    tries += 1
                    error_code = 'Obs coordinates included'
                    continue
                #elif ('[' + self.coord_template.format(*obs_traj[ped_idx, to_track, 0, ::-1], -self.past) in response
                #    or '(x4, y4)]' in response
                #    or ')]' not in response):
                #    tries += 1
                #    print('Prompt------------------------\n', prompt)
                #    print('Response------------------------\n', response)
                #    error_code = 'Invalid response shape'
                #    continue
                elif len(response) == 0:
                    tries += 1
                    error_code = 'Empty response'
                    continue
                
                # Convert to list, check validity
                #try:
                #    response_cleanup = re.sub('[^0-9()\{\},.:\-\n]', '', response)
                #    response_cleanup = [eval(line[1:]) for line in response_cleanup.split('\n') if len(line) > 20 and line.startswith('.{0') and line.endswith(')}')]
                #    response_cleanup = [[list(line[i]) for i in range(self.T)] for line in response_cleanup]
                try:
                    response_cleanup = re.sub('[^0-9()\[\],.\-\n]', '', response.replace(':', '\n')).replace('(', '[').replace(')', ']')
                    response_cleanup = [eval(line) for line in response_cleanup.split('\n') if len(line) > self.c and line.startswith('[[') and line.endswith(']]')]
                except:
                    tries += 1
                    error_code = 'Response to list failed'
                    continue
                #print('Response Clean------------------------\n', response_cleanup)
            
                # Remove repeated obs sequence or truncate the response
                if len(response_cleanup) >= self.c:
                    response_cleanup = response_cleanup[-self.c:]
                
                # Check validity
                if (len(response_cleanup) == self.c
                    and all(len(response_cleanup[i]) >= self.T for i in range(self.c))
                    and all(all(len(response_cleanup[i][j]) == 2 for j in range(self.T)) for i in range(self.c))):
                    # Add the response to the dump list
                    for i in range(self.c):
                        if len(response_cleanup[i]) >= self.T:
                            response_cleanup[i] = response_cleanup[i][:self.T]
                    response_cleanup = torch.as_tensor(response_cleanup, dtype=torch.float32)
                    llm_processed_list[ped_idx] = response_cleanup
                    
                    #llm_response_list[ped_idx, obj_idx] += response
                    break
                else:
                    tries += 1
                    error_code = 'Wrong response format'
                    continue
        
            torch.save(llm_processed_list, './predictions/llm_processed_list_scene_{}.pth'.format(ped_idx))

            llm_processed = llm_processed_list
            #llm_processed = torch.cat([llm_processed, torch.zeros((batch_size, self.c, self.T, 3), dtype=torch.float32, device=llm_processed.device)], dim=-1) # Pad features
        
        output = {}
        output['predicted_trajectory'] = llm_processed.to(batch['input_dict']['obj_trajs'].device)
        output['predicted_probability'] = torch.ones((batch_size, self.c), dtype=torch.float32, device=batch['input_dict']['obj_trajs'].device) / self.c

        return output, 0

    def configure_optimizers(self):
        return [], []