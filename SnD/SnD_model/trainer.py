
import threading
import torch
import os, shutil
from torch.utils.data import DataLoader
from SnD_datasets import PadBatchSeq, MixedCLSDataset, MixedSlotTaggingDataset, MixDataset
from tqdm import tqdm
import json
import torch.distributed as dist
import os, time, gc, json, pickle, argparse, math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import get_linear_schedule_with_warmup, Conv1D, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib
import copy
from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
import numpy as np

from SnD_model.utils import *
from SnD_model.utils_SnD import get_model_input, cal_prototype, semantic_drift_esitimation
from SnD_model.models import get_model, get_previous_model
from SnD_model.losses import compute_vae_loss, compute_lm_loss, KDLoss, JS_KDLoss

from info import TASK2INFO
from collections import OrderedDict

from metrics import compute_metrics 



def train_step(device, model, optimizer, loss_fn, beta, vae_total, lm_total, 
                                distill=False, only_decoder=False, only_vae=False, prev_model=None,
                                args=None, mem_std=None, global_step=0):
    output = []

    if only_decoder:
        vae_loss, vae_ce_loss, vae_kl_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    else:
        vae_loss, vae_ce_loss, vae_kl_loss = compute_vae_loss(
            device, model, loss_fn, beta, vae_total=vae_total, distill=distill, prev_model=prev_model, 
            args=args, mem_std=mem_std)

    lm_loss = compute_lm_loss(device, model, loss_fn, lm_total, distill=distill, prev_model=prev_model, args=args)          

    if not only_decoder and not only_vae:           
        total_loss = vae_loss + args.lm_loss_weight * lm_loss
    elif only_vae: 
        total_loss = vae_loss
    else:
        total_loss = lm_loss 

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  

    total_loss = total_loss / args.accumulate_steps
    total_loss.backward()


    if global_step % args.accumulate_steps ==0:
        optimizer.step()
        optimizer.zero_grad()
    output.append((vae_loss.item(), vae_ce_loss.mean().item(), vae_kl_loss.item(), lm_loss.item()))

    return output


class Trainer:
    def __init__(self, args, tokz, datasets, logger=None, cache_dir=None, memory=None, memory_std=None):
        self.args = args
        self.datasets = datasets
        self.logger = logger
        self.rank = self.args.local_rank
        self.device = self.args.device
        self.global_step = 0
        self.task_step = 0
        distributed = False if self.rank == -1 else True
        self.task_dict = {k:v for v,k in enumerate(self.args.tasks)}    

        if self.rank in [0, -1]:  
            self.train_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'train'), flush_secs=10)
            self.valid_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'valid'))
        
        self.config = GPT2Config.from_pretrained(self.args.gpt2_path)
        self.tokz = tokz
        print('len tokz gpt2',len(self.tokz),flush=True)

        self.cache_dir = cache_dir
        self.memory = memory 
        self.memory_std = memory_std 

       # * Load datasets. 
        self.data_loaders = {}
        for task in self.datasets:
            self.data_loaders[task] = {}
            train_sampler = torch.utils.data.RandomSampler(datasets[task]['train'])
            valid_sampler = None

            self.data_loaders[task]['train'] = DataLoader(datasets[task]['train'], batch_size=self.args.train_batch_size, 
                                            sampler=train_sampler, num_workers=self.args.num_workers, pin_memory=True, 
                                            collate_fn=PadBatchSeq(self.tokz.eos_token_id))
            self.data_loaders[task]['val'] = DataLoader(datasets[task]['val'], batch_size=self.args.eval_batch_size, 
                                            sampler=valid_sampler, num_workers=self.args.num_workers, pin_memory=True, 
                                            collate_fn=PadBatchSeq(self.tokz.eos_token_id))
            self.data_loaders[task]['test'] = DataLoader(datasets[task]['test'], batch_size=self.args.eval_batch_size, 
                                            sampler=valid_sampler, num_workers=self.args.num_workers, pin_memory=True, 
                                            collate_fn=PadBatchSeq(self.tokz.eos_token_id))
            self.data_loaders[task]['eval_train'] = DataLoader(datasets[task]['train'], batch_size=self.args.train_batch_size, 
                                            sampler=None, num_workers=self.args.num_workers, pin_memory=True, 
                                            collate_fn=PadBatchSeq(self.tokz.eos_token_id), shuffle=False)          
        np.random.seed(args.seed)
        prng = np.random.RandomState()
        torch.random.manual_seed(args.seed)
        gpu = not self.args.no_gpu
        if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

        self.beta = args.beta_0                         
        self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device, non_blocking=True)
        self.kd_loss = KDLoss(KD_term=self.args.KD_term, T=self.args.KD_temperature)

        self.save_folder = args.output_dir
        os.makedirs(self.save_folder, exist_ok=True) 
        self.output_encoder_dir = self.save_folder 
        self.res_file = os.path.join(self.save_folder, 'metrics.json')
        with open(self.res_file, 'w', encoding='utf-8') as f:                       
            pass

    def _train(self, model, curr_task, prev_tasks, all_tasks_res, prev_model=None):
        if model is None:
            assert len(prev_tasks) == 0
            model_path=self.args.gpt2_path
            VAE = get_model(self.config, self.args)
            VAE.initialize(model_path)
            self.logger.info("Successfully initialize the model with pretrained GPT2 model!")
        else:
            VAE = model
        VAE = VAE.to(self.device, non_blocking=True)    
        VAE.train()

        if self.args.add_kd:
            if prev_model is None:
                assert len(prev_tasks) == 0
                prev_VAE = get_previous_model(self.config, self.args)
                prev_VAE.initialize(model_path)
                prev_VAE.load_state_dict(VAE.state_dict())  
            else:
                prev_VAE = prev_model  
            prev_VAE.to(self.device)
            prev_VAE.eval()


        train_dataset = self.datasets[curr_task]['train']

        if self.args.gen_replay and len(prev_tasks)>0:
            pseudo_data_count = int(len(self.datasets[curr_task]['train']) * self.args.pseudo_data_ratio) // len(prev_tasks)
            if self.rank in [0, -1]:
                self.logger.info(f' pseudo data count for each task {pseudo_data_count}')

            inferred_CLS_data = {}
            inferred_ST_data = {}
            inferred_data = {}
            for task in prev_tasks:
                pseudo_output_file = os.path.join(self.args.output_dir, curr_task+'_pseudo_'+task+'.json')
                self.logger.info('Generated pseudo will be writen into '+str(pseudo_output_file))

                data = gen_pseudo_data_qa_once(VAE, task, self.datasets[task]['train'], max_output_len=-1,
                                    batch_size=self.args.eval_batch_size, target_count=pseudo_data_count,
                                    output_file=pseudo_output_file, 
                                    top_k=self.args.top_k, top_p=self.args.top_p, temperature=self.args.temperature,
                                    only_decoder=self.args.only_decoder, memory=self.memory, args=self.args,
                                    mem_std=self.memory_std, device=self.device, logger=self.logger)

                inferred_data[task] = data

                if self.rank in [0, -1]:   
                    self.logger.info(f'Inferring pseudo data from {task}')
                    for i in range(0, min(6, pseudo_data_count)):
                        self.logger.info(f' {data[i]}')

            prev_train_dataset = MixDataset(inferred_data, tokz=self.tokz, ctx_max_len=self.args.ctx_max_len)

        self.logger.info("Begin training!"+ "=" * 40)
        self.logger.info(f'Currently training {curr_task}'+'-'*10)

        evalout_dir = os.path.join(self.args.output_dir, curr_task)
        if os.path.exists(evalout_dir) and os.path.isdir(evalout_dir):
            shutil.rmtree(evalout_dir)
        os.makedirs(evalout_dir, exist_ok=True)  

        param_optimizer = list(VAE.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=self.args.lr, correct_bias=True)
        optimizer.zero_grad()
        if self.rank == -1:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)  
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) 
        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, sampler=train_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))
        if len(prev_tasks)>0:
            print('We have trained %d tasks'%(len(prev_tasks)),flush=True)
            prev_train_sampler = torch.utils.data.RandomSampler(prev_train_dataset)
            prev_train_loader = DataLoader(prev_train_dataset, batch_size=self.args.train_batch_size, sampler=prev_train_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))

        n_iter = self.args.num_train_epochs[curr_task] * len(train_loader)
        beta_list = frange_cycle_linear(n_iter, start=0.0, n_cycle=self.args.num_cycle, ratio=0.9)
        self.logger.info("Beta list we will use to train"+str(beta_list))

        t_total = len(train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs[curr_task]
        save_steps = int(t_total // self.args.num_train_epochs[curr_task] * self.args.save_epochs)
        if self.args.warmup_steps > 0:
            num_warmup_steps = self.args.warmup_steps
        else:
            num_warmup_steps = int(t_total * self.args.warmup_proportion)
        if self.args.eval_times_per_task > 0:
            eval_steps = int(t_total / self.args.eval_times_per_task)
        else:
            eval_steps = self.args.eval_steps
        if not self.args.nouse_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)


        for epoch in range(1, self.args.num_train_epochs[curr_task] + 1): 
            if self.rank in [0, -1]:
                self.logger.info(
                    f'num_warmup_steps: {num_warmup_steps}, t_total: {t_total}, save_steps: {save_steps}, eval_steps: {eval_steps}')

            self.task_step, iter_count, avg_loss = 0, 0, 0
            self.logger.info("Total epoches: %d" % self.args.num_train_epochs[curr_task])
            self.logger.info('*'*50+"Epoch: %d" % epoch+'*'*50)
            st = time.time() 

            ITER = tqdm(enumerate(train_loader), dynamic_ncols=True, total=len(train_loader))
            if len(prev_tasks) > 0:
                prev_ITER = enumerate(prev_train_loader)

            for i, data in ITER:
                optimizer.zero_grad() 
                beta = beta_list[i + (epoch-1) * len(ITER)]
                iter_count += 1

                if not self.args.general_prompt:
                    vae_total = get_model_input(data, input_type='vae', step=self.global_step, tokz=self.tokz, args=self.args)
                    lm_total = get_model_input(data, input_type='lm', step=self.global_step, tokz=self.tokz, args=self.args)

                else:
                    p_mask = data['gene_prompt_mask']  
                    p_tokens = data['gene_prompt_id'] 
                    px_tokens = data['gene_posterior_id']
                    px_mask = data['gene_posterior_mask']
                    input_tokens = data['gene_input_id'][..., :-1]
                    attention_mask = data['gene_input_mask'][...,:-1].contiguous()
                    input_label_mask = data['gene_input_label_mask'][...,1:].contiguous()
                    target_tokens = data['gene_input_id'][...,1:].contiguous()                    
                    lm_input_tokens = data['gene_all_id'][...,:-1]
                    lm_attention_mask = data['gene_all_mask'][...,:-1].contiguous()
                    lm_input_label_mask = data['gene_all_label_mask'][...,1:].contiguous()
                    lm_target_tokens = data['gene_all_id'][...,1:].contiguous()
                    lm_total = (lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens)
                    vae_total = (p_mask, p_tokens, px_mask, px_tokens, input_tokens, attention_mask, target_tokens, input_label_mask)

                all_output = train_step(self.device, VAE, optimizer, self.loss_fn, beta, 
                                    vae_total=vae_total, lm_total=lm_total, only_decoder=self.args.only_decoder, 
                                    only_vae=self.args.only_vae, distill=False, prev_model=None,
                                    args=self.args, mem_std=self.memory_std, global_step=self.global_step)
            
                vae_loss, vae_ce_loss, vae_kl_loss, lm_loss = all_output[-1]



                if self.args.add_kd:
                    if len(prev_tasks) > 0:     
                        if not self.args.general_prompt:
                            prev_data = next(iter(prev_train_loader))   
                            prev_vae_total = get_model_input(prev_data, input_type='vae', step=self.global_step, tokz=self.tokz, args=self.args)
                            prev_lm_total = get_model_input(prev_data, input_type='lm', step=self.global_step, tokz=self.tokz, args=self.args)
                        else:
                            prev_data = next(iter(prev_train_loader))
                            prev_lm_input_tokens = prev_data['gene_all_id'][...,:-1]
                            prev_lm_attn_mask = prev_data['gene_all_mask'][...,:-1].contiguous()
                            prev_lm_input_label_mask = prev_data['gene_all_label_mask'][...,1:].contiguous()
                            prev_lm_target_tokens = prev_data['gene_all_id'][...,1:].contiguous()
                            prev_p_mask = prev_data['gene_prompt_mask']
                            prev_p_tokens = prev_data['gene_prompt_id']
                            prev_px_tokens = prev_data['gene_posterior_id']
                            prev_px_mask = prev_data['gene_posterior_mask']
                            prev_input_tokens = prev_data['gene_input_id'][..., :-1]
                            prev_attention_mask = prev_data['gene_input_mask'][...,:-1].contiguous()
                            prev_input_label_mask = prev_data['gene_input_label_mask'][..., 1:].contiguous()
                            prev_target_tokens = prev_data['gene_input_id'][..., 1:].contiguous()                            

                            prev_vae_total = (prev_p_mask, prev_p_tokens, prev_px_mask, prev_px_tokens, prev_input_tokens, prev_attention_mask, 
                                              prev_target_tokens, prev_input_label_mask)
                            prev_lm_total = (prev_lm_input_tokens, prev_lm_attn_mask, prev_lm_input_label_mask, prev_lm_target_tokens)

                        _ = train_step(self.device, VAE, optimizer, self.kd_loss, beta, vae_total=prev_vae_total, lm_total=prev_lm_total, 
                                        only_vae=False, distill=True, prev_model=prev_VAE,                      
                                        args=self.args, mem_std=self.memory_std, global_step=self.global_step)

                optimizer.step()
                lr = scheduler.get_last_lr()[0]

                self.logger.info(f'EPOCH: {epoch}, task step: {self.task_step}, global step: {self.global_step} ' + \
                        'CVAE total loss: %.4f, CVAE reconstruction (ce) loss: %.4f, CVAE kl loss: %.4f, QA(lm) loss:  %.4f' \
                            % (vae_loss, vae_ce_loss, vae_kl_loss, lm_loss))


                self.train_writer.add_scalar('vae_loss_total', vae_loss, self.global_step)
                self.train_writer.add_scalar('vae_ce_loss', vae_ce_loss, self.global_step)
                self.train_writer.add_scalar('vae_kl_loss', vae_kl_loss, self.global_step)
                self.train_writer.add_scalar('lm_loss', lm_loss, self.global_step)
                self.train_writer.add_scalar('time_per_batch', time.time() - st, self.global_step)
                self.train_writer.add_scalar('lr', lr, self.global_step)
                self.train_writer.add_scalar('beta', beta, self.global_step)

                st = time.time()
                if not self.args.nouse_scheduler:                    
                    scheduler.step()

                self.global_step += 1 
                self.task_step += 1

                if self.global_step % self.args.eval_steps == 0 and self.args.eval_steps > 0:
                    self.do_eval(VAE, all_tasks_res, prev_tasks, curr_task, epoch)

            self.logger.info("Training loop. The %dth epoch completed."%epoch)


        task_save_folder = os.path.join(self.save_folder, str(curr_task))
        os.makedirs(task_save_folder, exist_ok=True)
        task_model_save_path = os.path.join(task_save_folder, 'model_'+'{:03d}'.format(self.global_step) + '.pt')
        torch.save(VAE.state_dict(), task_model_save_path)
        self.logger.info("Saving model checkpoint of %s to %s"%(curr_task, task_model_save_path))

        self.do_eval(VAE, all_tasks_res, prev_tasks, curr_task, epoch)

        if self.args.vae_type =='prototype_cvae':
            prototype_curtask = cal_prototype(model=VAE, dataloader=self.data_loaders[curr_task]['eval_train'],
                                    tokz=self.tokz, args=self.args, device=self.device)
            self.memory_std.memory[TASK2INFO[curr_task]['id']] = OrderedDict()
            self.memory_std.memory[TASK2INFO[curr_task]['id']]['name'] = curr_task
            self.memory_std.memory[TASK2INFO[curr_task]['id']].update(prototype_curtask) 
            VAE.train()  

        if self.args.vae_type =='prototype_cvae' and self.args.semantic_drift_flag and len(prev_tasks)>0:
            for pre_task in prev_tasks:
                mean_semantic_drift, logvar_semantic_drift = semantic_drift_esitimation(VAE, prev_VAE, 
                                                    dataloader = self.data_loaders[curr_task]['eval_train'], 
                                                    tokz=self.tokz, mem_std = self.memory_std,
                                                    args=self.args, device=self.device, pre_task=pre_task)
                self.memory_std.memory[TASK2INFO[pre_task]['id']]['mean_drift'].append(mean_semantic_drift)
                self.memory_std.memory[TASK2INFO[pre_task]['id']]['logvar_drift'].append(logvar_semantic_drift)
            VAE.train()  
            prev_VAE.eval()  
    
        task_final_save_path = os.path.join(self.save_folder, str(curr_task)+'_model'+'.pt')
        model_save_path = os.path.join(self.save_folder, 'model_last.pt')
        torch.save(VAE.state_dict(), model_save_path)
        self.logger.info('Saving total model checkpoint to %s', model_save_path)
        self.logger.info('='*25+'Training complete!'+'='*25)

        if self.args.add_kd:
            return VAE, prev_VAE, all_tasks_res
        return VAE, all_tasks_res


    def do_eval(self, VAE, all_tasks_res, prev_tasks, curr_task, epoch):
        all_res_per_eval = {}
        avg_metric_loss, avg_metric = [], []
        for val_task in self.args.tasks[:len(prev_tasks)+1]:
            eval_loss = self._eval(VAE, val_task, self.kd_loss)                
            avg_metric_loss.append(eval_loss['lm_loss'])
            all_prior_info = None
            predictions, references = self.get_answers_for_eval(VAE, val_task, all_prior_info=all_prior_info, 
                        out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None)
            eval_metric = compute_metrics(predictions, references, xlingual=False, task_name=val_task)
            avg_metric.append(eval_metric[TASK2INFO[val_task]['metric']])

            for key, value in eval_loss.items():
                self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)
            for key, value in eval_metric.items():
                self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)

            self.logger.info('Evaluation! '+f'Current task: {curr_task}, task_epoch: {epoch}, task step: {self.task_step}, ' + \
                f'global step: {self.global_step}, {val_task}: eval loss: {eval_loss}, eval metric: {eval_metric}')
            
            all_res_per_eval[val_task] = eval_metric

        all_tasks_res.append(all_res_per_eval)

        with open(self.res_file, 'a+', encoding='utf-8') as f:
            print(json.dumps(all_res_per_eval,ensure_ascii=False), file=f)

        self.valid_writer.add_scalar(f'avg/loss', sum(avg_metric_loss)/len(avg_metric_loss), self.global_step)
        self.valid_writer.add_scalar(f'avg/metric', sum(avg_metric)/len(avg_metric), self.global_step)

    def _eval(self, model, task, loss_fn):
        lm_recorder = MetricsRecorder(self.device, 'lm_loss')

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(self.data_loaders[task]['val']):
                bs = data['all_id'].shape[0]
                lm_total = get_model_input(data, input_type='lm', step=100, tokz=self.tokz, 
                                                            args=self.args)
                loss = compute_lm_loss(self.device, model, loss_fn, lm_total)
                lm_recorder.metric_update(i, {'lm_loss': loss})

        if self.rank != -1:
            lm_recorder.all_reduce()
        return lm_recorder

    def train(self, tasks, continue_from_task=None):
        model = None
        prev_model = None
        self.logger.info('Total tasks number is '+str(len(tasks)))

        all_tasks_res = []
        res_file = os.path.join(self.save_folder, 'metrics.json')

        continue_task_index = 0
        if continue_from_task!=None:
            model, prev_model, continue_task_index, all_tasks_res = self.get_cl_checkpoint(tasks, continue_from_task)

        for i in range(continue_task_index, len(tasks)):
            if self.args.add_kd:
                model, prev_model, all_tasks_res = self._train(
                    model, prev_model=prev_model, curr_task=tasks[i], prev_tasks=tasks[:i], 
                    all_tasks_res=all_tasks_res)
            else:
                model, all_tasks_res = self._train(
                    model, curr_task=tasks[i], prev_tasks=tasks[:i], all_tasks_res=all_tasks_res)

            if self.args.add_kd:
                prev_model.load_state_dict(model.state_dict())
             
            self.logger.info('We have trained %d tasks'%(i+1))
            if self.args.use_memory:
                self.logger.info('Memory has saved information of %d tasks'%
                                                        (len(self.memory.memory.keys()))+'-'*10)

        if self.args.use_memory:
            save_memory_path = os.path.join(self.args.memory_path, 'memory.pt')
            torch.save(self.memory, save_memory_path)

    def get_cl_checkpoint(self, tasks, continue_from_task):

        continue_task_index = tasks.index(continue_from_task)

        model_path=self.args.gpt2_path
        VAE = get_model(self.config, self.args)
        prev_VAE = get_previous_model(self.config, self.args)
        VAE.initialize(model_path)
        prev_VAE.initialize(model_path)
        VAE.to(self.device)
        prev_VAE.to(self.device)
        state_dict = torch.load(os.path.join(self.save_folder, 'model_last.pt'), map_location=self.device)
        VAE.load_state_dict(state_dict)
        prev_VAE.load_state_dict(state_dict)
        VAE.train()
        prev_VAE.eval()

        if self.args.vae_type =='prototype_cvae':
            self.memory_std.load(
                    os.path.join(self.save_folder, str(tasks[continue_task_index-1])+'_mem_std'+'.pt'))

        if os.path.exists( os.path.join(self.save_folder, 'metrics.json')):
            with open(os.path.join(self.save_folder, 'metrics.json'), "r") as f:
                all_tasks_res = [json.loads(row) for row in f.readlines()]
        else:
            all_tasks_res = []

        return VAE, prev_VAE, continue_task_index, all_tasks_res




    def get_answers_for_eval(self, model, task, out_dir=None, all_prior_info=None):
        max_ans_len = max(self.datasets[task]['train'].max_ans_len, 
                                    self.datasets[task]['val'].max_ans_len, 
                                    self.datasets[task]['test'].max_ans_len) + 1
        if out_dir is not None:
            out_dir = os.path.join(out_dir, f'{task}_step{self.global_step}.json')
            out_dir_content = []
            
        with torch.no_grad():
            model.eval()
            pred_ans_all, gold_ans_all = [], []
            context_all = []
            all_pred_task_name = []
            
            for i, data in enumerate(self.data_loaders[task]['val']):
                if not self.args.classIL: 
                    example = data['context_id'].to(self.device, non_blocking=True)
                    example_lens = data['context_lens'].to(self.device, non_blocking=True)
                elif self.general_prompt:
                    example = data['general_context_id'].to(self.device, non_blocking=True)
                    example_lens = data['general_context_lens'].to(self.device, non_blocking=True)
                   
                if out_dir is not None:
                    context_all.append(example)
                gold_ans = data['ans_id'].to(self.device, non_blocking=True)

                pred_ans = get_answer(self.tokz, model, example, example_lens, max_ans_len, sampling=False, args=self.args)

                pred_ans_all.append(pred_ans)
                gold_ans_all.append(gold_ans)
                
            if self.args.classIL:           
                right_pred_task = [1 for pred_task in all_pred_task_name if pred_task==task] 
                self.logger.info('Correct prediction of task name ratio is: '+str(len(right_pred_task)/len(all_pred_task_name)))

            pred_ans_all = communicate_tensor(pred_ans_all, pad_token=self.tokz.eos_token_id).tolist()
            gold_ans_all = communicate_tensor(gold_ans_all,pad_token=self.tokz.eos_token_id).tolist()
            if out_dir is not None:
                context_all = communicate_tensor(context_all,pad_token=self.tokz.eos_token_id).tolist()

            predictions = [self.tokz.decode(cut_eos(l, self.tokz.eos_token_id)) for l in pred_ans_all]
            references = [self.tokz.decode(cut_eos(l, self.tokz.eos_token_id)) for l in gold_ans_all]

            for i in range(len(pred_ans_all)):
                res = {}
                res['context'] = self.tokz.decode(strip_list(context_all[i], self.tokz.eos_token_id))
                res['ans_gold'] = self.tokz.decode(cut_eos(gold_ans_all[i], self.tokz.eos_token_id))
                res['ans_pred'] = self.tokz.decode(cut_eos(pred_ans_all[i], self.tokz.eos_token_id))
                out_dir_content.append(res)

        model.train()

        if out_dir is not None:
            with open(out_dir, 'w', encoding='utf-8') as outfile:
                for res in out_dir_content:
                    print(json.dumps(res), file=outfile)

        return predictions, references 

