import numpy as np
import torch
import torch.nn as nn
from scipy import special
from torch.distributions import MultivariateNormal, Laplace

from unitraj.models.base_model.base_model import BaseModel
import transformers
import os
from tqdm import tqdm
import re

max_timeout = 20
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
            model_kwargs={"torch_dtype": torch.float16},
            device="cuda"
        )

        self.prompt_system = config['sys_prompt']
        self.prompt_template_list = config['prompts']
        self.coord_template = "({0:.2f}, {1:.2f})"
        self.scene_name_template = "scene_{:04d}"
        

        self.criterion = Criterion(self.config)

    def forward(self, batch):
        inputs = batch['input_dict']

        num_scenes = inputs['obj_trajs'].shape[0]

        for scene_idx in (progressbar:=tqdm(range(num_scenes), desc='Chatbot started!')):

            obs_traj = inputs['obj_trajs'][scene_idx,..., :2].cpu().numpy()
            pred_traj = inputs['center_gt_trajs'][scene_idx,..., :2].cpu().numpy()
            num_ped, obs_len, _ = obs_traj.shape
            pred_len, _ = pred_traj.shape

            llm_response_list = [["" for _ in range(len(self.prompt_template_list))] for _ in range(num_ped)]
            llm_processed_list = [[] for _ in range(num_ped)]
        
            for ped_idx in range(num_ped):
                messages = [{"role": "system", "content": self.prompt_system.format(obs_len, pred_len, self.c)}]

                for prompt_idx in range(len(self.prompt_template_list)):
                    coord_str = '[' + ', '.join([self.coord_template.format(*obs_traj[ped_idx, i]) for i in range(obs_len)]) + ']'
                    prompt = self.prompt_template_list[prompt_idx].format(obs_len, pred_len, coord_str, self.c)
                    messages.append({"role": "user", "content": prompt})
                    
                    error_code = ''
                    timeout = 0
                    add_info = 0

                    while timeout < max_timeout:
                        # Prevent RateLimitError by waiting for 20 seconds
                        progressbar.set_description('Ped_id {}/{} Prompt_no. {}/{} retry {}/{} {}'.format(ped_idx+1, num_ped, prompt_idx+1, len(self.prompt_template_list), timeout, max_timeout, error_code))
                        
                        # Set additional information and settings when it kept failing
                        tmp = 1.0 if timeout >= max_timeout // 2 else temperature
                        if prompt_idx == 0 and timeout == max_timeout // 4 and add_info < 1:
                            messages[-1]['content'] += f'\nProvide {self.c:d} hypothetical scenarios based on different extrapolation methods.'
                            add_info = 1
                        elif prompt_idx == 0 and timeout == (max_timeout // 4) * 2 and add_info < 2:
                            messages[-1]['content'] += '\nYou can use methods like linear interpolation, polynomial fitting, moving average, and more.'
                            add_info = 2
                        
                        # Run the Chatbot model
                        try:
                            response = self.pipeline(
                                messages,
                                max_new_tokens=self.max_new_tokens,
                                temperature=tmp,
                            )
                            response = response[0]["generated_text"][-1]['content']
                            # print('Coord_string--------------------\n', coord_str)
                            # print('Response------------------------\n', response)
                        except Exception as err:
                            error_code = f"Unexpected {err=}, {type(err)=}"
                            print(error_code)
                            response = ''
                            timeout += 1
                            continue

                        # filter out wrong answers
                        if (not (abs(obs_traj[ped_idx, 0] - obs_traj[ped_idx, -1]).sum() < 0.3
                            or abs(obs_traj[ped_idx, 0] - obs_traj[ped_idx, 2]).sum() < 0.2
                            or abs(obs_traj[ped_idx, -3] - obs_traj[ped_idx, -1]).sum() < 0.2) and '[' + self.coord_template.format(*obs_traj[ped_idx, 0]) in response):
                            if prompt_idx == 0:
                                timeout += 1
                                error_code = 'Obs coordinates included'
                                continue
                        elif ('[' + self.coord_template.format(*obs_traj[ped_idx, 0, ::-1]) in response
                            or '(x4, y4)]' in response
                            or ')]' not in response):
                            timeout += 1
                            error_code = 'Invalid response shape'
                            continue
                        elif len(response) == 0:
                            timeout += 1
                            error_code = 'Empty response'
                            continue
                        
                        # Convert to list, check validity
                        try:
                            response_cleanup = re.sub('[^0-9()\[\],.\-\n]', '', response.replace(':', '\n')).replace('(', '[').replace(')', ']')
                            response_cleanup = [eval(line) for line in response_cleanup.split('\n') if len(line) > 20 and line.startswith('[[') and line.endswith(']]')]
                        except:
                            timeout += 1
                            error_code = 'Response to list failed'
                            continue
                        
                        # Remove repeated obs sequence or truncate the response
                        if len(response_cleanup) >= self.c:
                            response_cleanup = response_cleanup[-self.c:]
                        
                        # Check validity
                        if (len(response_cleanup) == self.c
                            and all(len(response_cleanup[i]) == pred_len for i in range(self.c))
                            and all(all(len(response_cleanup[i][j]) == 2 for j in range(pred_len)) for i in range(self.c))):
                            # Add the response to the dump list
                            llm_processed_list[ped_idx].extend(response_cleanup)
                            llm_response_list[ped_idx][prompt_idx] = response
                            messages.append({"role": "assistant", "content": response})
                            break
                        else:
                            timeout += 1
                            error_code = 'Wrong response format'
                            continue

                    if timeout >= max_timeout:
                        print("Chatbot Timeout! Error scene_idx: {} ped_idx: {} prompt_idx: {}".format(scene_idx, ped_idx, prompt_idx))
                        break


        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)

        
        output = {}
        output['predicted_probability'] = torch.rand(ground_truth.shape[0], self.c)  # [B, c]
        output['predicted_trajectory'] = torch.rand(ground_truth.shape[0], self.c, ground_truth.shape[1], 5)  # [B, c, T, 5] 

        loss = self.criterion(output, ground_truth)

        return output, loss

    def configure_optimizers(self):
        return [], []


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt):
        return self.nll_loss_multimodes(out, gt)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        cov = torch.zeros((B, T, 2, 2)).to(pred.device)
        cov[:, :, 0, 0] = sigma_x ** 2
        cov[:, :, 1, 1] = sigma_y ** 2
        cov[:, :, 0, 1] = rho * sigma_x * sigma_y
        cov[:, :, 1, 0] = rho * sigma_x * sigma_y

        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4])

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
        log_lik = np.zeros((batch_sz, modes))
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik[:, kk] = -nll.cpu().numpy()

        priors = modes_pred.detach().cpu().numpy()
        log_posterior_unnorm = log_lik + np.log(priors)
        log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
        post_pr = np.exp(log_posterior)
        post_pr = torch.tensor(post_pr).float().to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
