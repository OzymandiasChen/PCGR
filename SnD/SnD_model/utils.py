import logging
import random
import torch 
import numpy as np
from torch.utils.data import DataLoader
from SnD_datasets import PadBatchSeq, pad_seq
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import os, time, gc, json, pickle, argparse, math, re
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F

from info import TASK2INFO

from SnD_model.utils_SnD import get_prototype



loggers = {}
all_slot_dict = json.load(open('slot_label_dict.json'))

def get_logger(filename, level=logging.INFO, print2screen=True):
    global loggers
    import logging

    if os.path.exists(filename):
        os.remove(filename)

    if loggers.get(filename):
        return loggers.get(filename)
    else:
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fh = logging.FileHandler(filename, encoding='utf-8')
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] >> %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        if print2screen:
            logger.addHandler(ch)
        loggers[filename] = logger
        return logger

def frange_cycle_linear(n_iter, start=0.01, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) 

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """
    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f

def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f


def get_mask(x_len):
    mask = torch.arange(max(x_len), device=x_len.device)[None, :] < x_len[:, None]  
    return mask.bool()

def get_reverse_mask(x_len):
    mask = torch.arange(max(x_len)-1, -1, -1, device=x_len.device)[None, :] < x_len[:, None] 
    return mask.bool()

def compare_tokens(x, y, eos_id):
    if eos_id in x:
        x = x[:x.index(eos_id)]
    if eos_id in y:
        y = y[:y.index(eos_id)]
    return x == y

def pad_tensor(tensor, length, pad_token=0):
    return torch.cat([tensor, tensor.new(tensor.size(0), length - tensor.size()[1]).fill_(pad_token)], dim=1)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)

def communicate_tensor(tensor_list, pad_token=0):
    '''
    collect tensors from all processes
    # collect results from different nodes                  # list of tensor to a single tensor;
    '''
    if len(tensor_list) == 0:
        return None
    device = tensor_list[0].device
    max_len = torch.tensor(max([i.shape[1] for i in tensor_list]), dtype=torch.int64, device=device)
    if dist.is_initialized():  
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    tensor = torch.cat([pad_tensor(i, max_len, pad_token) for i in tensor_list], dim=0)
    tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    max_tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    if dist.is_initialized():                               
        dist.all_reduce(max_tensor_bs, op=dist.ReduceOp.MAX)  
        if max_tensor_bs != tensor_bs:
            tensor = torch.cat([tensor, tensor.new(max_tensor_bs-tensor_bs, tensor.shape[1]).fill_(pad_token)], dim=0)

        tensor_list = [torch.ones_like(tensor).fill_(pad_token) for _ in range(dist.get_world_size())]
        tensor_bs_list = [torch.ones_like(tensor_bs).fill_(pad_token) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        dist.all_gather(tensor_list=tensor_bs_list, tensor=tensor_bs)
        for i in range(dist.get_world_size()):
            tensor_list[i] = tensor_list[i][:tensor_bs_list[i]]
        tensor = torch.cat(tensor_list, dim=0)
    return tensor

class MetricsRecorder(object):
    def __init__(self, device, *metric_names):
        self.metric_names = list(metric_names)
        self.device = device
        self.metrics = {}
        for metric_name in metric_names:
            self.metrics[metric_name] = torch.tensor(0, dtype=torch.float64, device=self.device)

    def metric_update(self, batch_no, metric_values_dict):
        for k, v in metric_values_dict.items():
            self.metrics[k] = (self.metrics[k] * batch_no + v) / (batch_no + 1)

    def add_to_writer(self, writer, step, group):
        for n in self.metric_names:
            m = self.metrics[n].item()
            writer.add_scalar('%s/%s' % (group, n), m, step)

    def write_to_logger(self, logger, epoch, step):
        log_str = 'epoch {:>3}, step {}'.format(epoch, step)
        for n in self.metric_names:
            m = self.metrics[n].item()
            log_str += ', %s %g' % (n, m)
        logger.info(log_str)

    def items(self):
        return self.metrics.items()

    def all_reduce(self):
        for n in self.metric_names:
            torch.distributed.all_reduce(self.metrics[n], op=torch.distributed.ReduceOp.SUM)
            self.metrics[n] /= torch.distributed.get_world_size()

    def __getitem__(self, k):
        return self.metrics[k]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __repr__(self):
        return self.metrics.__repr__()
    
    def __str__(self):
        return self.metrics.__str__()

    def keys(self):
        return self.metrics.keys()


def cut_eos(seq, eos_id):
    if eos_id not in seq:
        return seq
    return seq[:seq.index(eos_id)]
    

def infer_model_pred(model, tokz, dataset, outfile, batch_size=30):
    max_ans_len = dataset.max_ans_len + 1
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=3, pin_memory=True, collate_fn=PadBatchSeq(0))
    device = model.device
    with open(outfile, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(data_loader):
                bs = data['context_id'].shape[0]
                context = data['context_id'].to(device, non_blocking=True)
                context_lens = data['context_lens'].to(device, non_blocking=True)
                mask = get_reverse_mask(context_lens)
    
                output_sequence = model.generate(
                    input_ids=context, attention_mask=mask, do_sample=False, eos_token_id=tokz.eos_token_id,
                    pad_token_id=tokz.eos_token_id, max_length=context.shape[1] + max_ans_len, early_stopping=True)

                cls_res = output_sequence[:,context.shape[1]:].tolist()
                ans = data['ans_id'].tolist()
                
                for i in range(bs):
                    res = {}
                    res['context'] = tokz.decode(context[i][-context_lens[i]:])
                    res['ans_gold'] = tokz.decode(ans[i][:data['ans_lens'][i]-1])
                    res['ans_pred'] = tokz.decode(cut_eos(cls_res[i], tokz.eos_token_id))
                    print(json.dumps(res), file=f)


def cal_metrics_from_pred_files(res_file):
    with open(res_file, 'r', encoding='utf-8') as f:
        res = [json.loads(i) for i in f.readlines()]
    y_true = [i['ans_gold'] for i in res]
    y_pred = [i['ans_pred'] for i in res]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
    }

def slot_f1_score(pred_slots, true_slots):

    slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
    slot_type_f1_scores = []

    for slot_type in slot_types:           
        predictions_for_slot = [[p for p in prediction if slot_type in p] for prediction in pred_slots] 
        labels_for_slot = [[l for l in label if slot_type in l] for label in true_slots]

        proposal_made = [len(p) > 0 for p in predictions_for_slot]
        has_label = [len(l) > 0 for l in labels_for_slot]
        prediction_correct = [prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)]
        true_positives = sum([
            int(proposed and correct)
            for proposed, correct in zip(proposal_made, prediction_correct)])

        num_predicted = sum([int(proposed) for proposed in proposal_made])
        num_to_recall = sum([int(hl) for hl in has_label])

        precision = true_positives / (1e-5 + num_predicted)
        recall = true_positives / (1e-5 + num_to_recall)

        f1_score = 2 * precision * recall / (1e-5 + precision + recall)
        slot_type_f1_scores.append(f1_score)

    return np.round(np.mean(slot_type_f1_scores), 3)       




def get_answer(tokz, lm_model, example, example_lens, max_ans_len, sampling=False, args=None):
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    lm_model.eval()
    device = 'cuda'

    if example_lens==None:
        mask=None
    else:
        mask = get_reverse_mask(example_lens).to(device)
    eos_token = tokz.eos_token_id
    pad_token = tokz.eos_token_id

    output_seq = lm_model.decoder.generate(input_ids=example, attention_mask=mask, 
                                do_sample=False, eos_token_id=eos_token, pad_token_id=pad_token, 
                                max_length=example.shape[1]+max_ans_len, early_stopping=True)
    return output_seq[:, example.shape[1]:]

def textid_decode(text, eos, tokz):
    if eos in text:
        text = text[:text.index(eos)] 
    text = tokz.decode(text).strip()
    return text

def padding_convert(text_list, eos):
    tt_list = []
    for text in text_list:
        if eos in text:
            eos_indexs = [i for i, x in enumerate(text) if x==eos]
            if len(eos_indexs)>1: text = text[:eos_indexs[1]]       # one eos is enough, why two?
        tt_list.append(text)
    tt_lens = [len(i) for i in tt_list]
    tt_lens = torch.tensor(tt_lens, dtype=torch.long).to('cuda')
    tt_pad = torch.tensor([pad_seq(i, eos, max(tt_lens), pad_left=True) for i in tt_list], dtype=torch.long).to('cuda')
    return tt_pad, tt_lens



def sample_sequence(model, tokenizer, length, batch_size=None, p_mask=None, p_tokens=None, p_lens=None, 
                    temperature=1, top_k=100, top_p=0.95,  sampling=True, only_decoder=False, memory=None, 
                    args=None, task=None, use_prior=False,
                    mem_std=None, task_ids=None, device=None):
    p_tokens = p_tokens.to(device)
    
    if p_mask is not None: p_mask = p_mask.to(device) 
    eos_token = tokenizer.eos_token_id

    with torch.no_grad():
        if not only_decoder:
            if memory is None:
                if args.vae_type == 'prototype_cvae':
                    prototype = get_prototype(mem_std=mem_std, task_ids=task_ids, args=args, device=device)
                    z_proj = model(prototype=prototype, phase='inference')[0]
            elif use_prior:
                if args.save_z:
                    z_proj = memory.memory[task][1]['prior_z']
                else:
                    old_prior_mean, old_prior_logvar = memory.memory[task][1]['prior']
                    z = model.reparameterize(old_prior_mean, old_prior_logvar)
                    z_proj = model.latent_mlp(z) * args.alpha_z
                    assert not torch.isnan(z).any(), 'training get nan z'
            else: 
                if args.save_z:
                    z_proj = random.choice(memory.memory[task][1]['posterior_z'])
                else:
                    prev_post_mean, prev_post_logvar = random.choice(memory.memory[task][1]['posterior'])
                    z = model.reparameterize(prev_post_mean, prev_post_logvar)
                    z_proj = model.latent_mlp(z) * args.alpha_z
                    assert not torch.isnan(z).any(), 'training get nan z'
        else:
            z_proj = None

        model_kwargs = {'latent_proj':z_proj} # ! 
        output_seq = model.decoder.generate(input_ids=p_tokens, attention_mask=p_mask, do_sample=True, 
                        eos_token_id=eos_token, pad_token_id=eos_token, max_length=length, early_stopping=True, 
                        **model_kwargs)
        
    return output_seq





def gen_pseudo_data_qa_once(model, task, dataset, max_output_len=90, batch_size=30, target_count=100, output_file=None, 
                    top_k=100, top_p=0.95, temperature=1, only_decoder=False, memory=None, args=None, 
                    mem_std=None, device=None, logger=None):
    if args.max_sample_len_flag:
        max_q_len=dataset.max_q_len
        max_ans_len=dataset.max_ans_len
        max_output_len = max_q_len+ max_ans_len
    else:
        max_q_len=96
        max_ans_len=100
        max_output_len = max_q_len+ max_ans_len
    prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id
    prompt_mask, prompt_lens = None, None
    ans_prompt_id_ls = dataset.pseudo_ans_prompt_id
    max_output_len += len(prompt_id) + len(ans_prompt_id_ls)

    prompt_id = torch.LongTensor([prompt_id for _ in range(batch_size)]).to(device)
    ans_prompt_id = torch.LongTensor([ans_prompt_id_ls for _ in range(batch_size)]).to(device)
    task_ids = torch.LongTensor([TASK2INFO[task]['id'] for _ in range(batch_size)]).to(device)

    pseudo_list = []
    utter_set = set()
    eos_token = dataset.tokz.eos_token_id

    if output_file is None:
        raise ValueError("Pseudo output file is not specified.")
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)    

    batch_cnt=0
    while len(pseudo_list) < target_count:
        batch_cnt+=1
        if len(pseudo_list) <= target_count // 5:
            use_prior = True
        else: use_prior = False       
        with torch.no_grad():
            model.eval()
            output_seq = sample_sequence(model, dataset.tokz, length=max_output_len, batch_size=batch_size,
                        p_tokens=prompt_id, p_mask=prompt_mask, p_lens=prompt_lens, temperature=temperature,
                        top_k=top_k, top_p=top_p, sampling=True, only_decoder=only_decoder, memory=memory, 
                        args=args, task=task, use_prior=use_prior,
                        mem_std=mem_std, task_ids=task_ids, device=device)
        output_list = output_seq.tolist()

        for i in range(batch_size):
            output_id = output_list[i][1:]
            if eos_token in output_id:
                output_id = output_id[:output_id.index(eos_token)]
            output = dataset.tokz.decode(output_id)             
            if ' "? Answer: ' in output:
                match_oj = re.match(r'(.+?) task, {} " (.+?) "\? Answer: (.+)'.format(TASK2INFO[task]['part_prompt']), output, re.M|re.I)
                if match_oj!= None:
                    utter = match_oj.group(2)
                    label = match_oj.group(3)
                else: continue

                if TASK2INFO[task]['task_type'] == 'intent' and args.check_during_pseudo_gene_flag:
                    if not label.lower().strip() in dataset.ans_set:
                        continue
                if TASK2INFO[task]['task_type'] == 'slot' and 'No slot in this sentence.' in label and args.check_during_pseudo_gene_flag:
                    continue

            else: continue

            res = {'task_name': task,'utter': utter, 'label': label} 
               
            if res is not None:
                utter = res['utter']
                label = res['label']
                print('UTTER::', utter,'====>> LABEL::', label, flush=True) 
                if utter not in utter_set and res['task_name']==task and label!='':
                    if TASK2INFO[task]['task_type'] == 'slot':
                        select = True
                        if select:                            
                            utter_set.add(utter)
                            if label[-1] == ';': label = label[:-1]
                            pseudo_list.append([utter,label])
                    else: 
                        utter_set.add(utter)
                        pseudo_list.append([utter, label])
    pseudo_list = pseudo_list[:target_count]   

    with open(output_file, 'w', encoding='utf8') as f:
        for utter, label in pseudo_list:
            print(json.dumps({'Utterence': utter, 'Label': label}, ensure_ascii=False), file=f)

    return pseudo_list



def strip_list(seq, eos_id):
    l, r = 0, len(seq)-1
    for i in range(len(seq)):
        if seq[i] != eos_id:
            break
        l = i
    for i in range(len(seq)-1, -1, -1):
        if seq[i] != eos_id:
            break
        r = i
    return seq[l+1:r]
  

def get_all_priors(model, tokz, args):
    def pseudo_prompt(task):
        return f"In the \"{task}\" task, which intent category best describes: \""
    all_prior_info = {}
    for task in args.tasks:
        prompt_id = [tokz.bos_token_id]+tokz.encode(pseudo_prompt(task))+[tokz.eos_token_id]
        prompt_id = torch.LongTensor(prompt_id).to('cuda')
        prior_out = model.encoder(input_ids=prompt_id)
        prior_emb, _ = model.avg_attn(prior_out[0])
        prior_mean, prior_logvar = model.prior_mean(prior_emb), model.prior_logvar(prior_emb)
        all_prior_info[task]=(prior_mean, prior_logvar)
    return all_prior_info

def get_nearest_task(model, tokz, sample, all_prior_info, args):
    def pseudo_prompt(task):
        return f"In the \"{task}\" task, which intent category best describes: \""
    all_posteriors={} 
    batch_size = len(sample['utter_id'])
    for task in args.tasks:
        prompt_id = [tokz.bos_token_id]+tokz.encode(pseudo_prompt(task))
        bt_prompt_id = torch.LongTensor([prompt_id for _ in range(batch_size)]).to('cuda') 
        bt_px_id = torch.cat((bt_prompt_id,sample['utter_id'].to('cuda')),dim=1)
        bt_px_id = bt_px_id.to('cuda')
        if len(bt_px_id)!=batch_size:
            raise ValueError('Tensor concatenate is wrong.')

        post_out = model.encoder(input_ids=bt_px_id)
        post_emb, _ = model.avg_attn(post_out[0])
        post_mean, post_logvar = model.post_mean(post_emb), model.post_logvar(post_emb)
        all_posteriors[task]=(post_mean, post_logvar)

    min_kl = 1e10
    res_task = args.tasks[0]
    all_kl_dist = []
    for task in all_prior_info.keys():
        prior_mean, prior_logvar = all_prior_info[task]
        post_mean, post_logvar = all_posteriors[task]
        kl_dist = kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
        all_kl_dist.append(kl_dist)
        if kl_dist < min_kl:
            min_kl = kl_dist
            res_task = task
    print(all_kl_dist,flush=True)
    return res_task

def get_pred_context(tokz, pred_task_name, gt_task_name, sample): 
    new_list = []
    for ss in sample['context_id'].tolist(): 
        context = tokz.decode(ss) 
        new_context = re.sub(gt_task_name,pred_task_name,context)
        new_context_id = tokz.encode(new_context)
        
        new_list.append(new_context_id)
    context_lens = [len(i) for i in new_list]
    context_mask = torch.ByteTensor(
        [[1] * context_lens[i] + [0] * (max(context_lens)-context_lens[i]) for i in range(len(context_lens))]) 
    new_res = torch.tensor(
        [pad_seq(i, tokz.eos_token_id, max(context_lens), pad_left=True) for i in new_list], dtype=torch.long
        ).to('cuda')
    new_lens = torch.tensor(context_lens,dtype=torch.long).to('cuda')
    return new_res, new_lens

def slot_select_pseudo(utter, answer, task_name):
    slot_list = all_slot_dict[task_name]
    pair_list = answer.split('; ')
    pseudo_slot = []
    if len(pair_list) == 0:
        return False
    for pair in pair_list: 
        slot_value = pair.split(': ')
        if len(slot_value) != 2:
            return False
        slot, value = slot_value
        if slot not in slot_list or value not in utter or value == '':
            return False
    return True




