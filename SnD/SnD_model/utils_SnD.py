
import torch
import torch.distributed as dist

from info import TASK2INFO


epsilon=1e-8


def reparameterize(mu, logvar, nsamples=1):      
    batch_size, nz = mu.size()
    std = logvar.mul(0.5).exp()
    mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
    std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
    eps = torch.zeros_like(std_expd).normal_()
    return mu_expd + torch.mul(eps, std_expd)


def get_prototype(mem_std, task_ids, args, device):
    prototype_mean = torch.stack([mem_std.memory[task_id.item()]['mean'] for task_id in task_ids], dim=0)
    prototype_logvar = torch.stack([mem_std.memory[task_id.item()]['logvar'] for task_id in task_ids], dim=0)
    if args.semantic_drift_flag:
        semantic_drift_list = [mem_std.memory[task_id.item()]['mean_drift'] for task_id in task_ids]
        semantic_drift = torch.stack([torch.sum(torch.stack(s_d_l), dim=0) for s_d_l in semantic_drift_list], dim=0)
        prototype_mean = prototype_mean + args.semantic_drift_weight * semantic_drift
        if args.logvar_drift_flag:
            logvar_semantic_drift_list = [mem_std.memory[task_id.item()]['logvar_drift'] for task_id in task_ids]
            logvar_semantic_drift = torch.stack([torch.sum(torch.stack(s_d_l), dim=0) for s_d_l in logvar_semantic_drift_list], dim=0)
            prototype_logvar = prototype_logvar + args.logvar_semantic_drift_weight * logvar_semantic_drift
    if args.self_arguement_flag:
        prototype = args.self_arguement_weight * reparameterize(prototype_mean, prototype_logvar)
        prototype = prototype.squeeze(dim=1)
    else:
        prototype = prototype_mean
    prototype = prototype.to(device)
    return prototype



def cal_prototype(model, dataloader, tokz, args, device):
    post_emb_all = get_all_px_embeddings(model, dataloader, tokz, args, device)
    post_mean = torch.mean(post_emb_all, dim=0)
    post_logvar = torch.log(torch.var(post_emb_all, dim=0, unbiased=False)+1e-8)
    return { 'mean': post_mean, 'logvar': post_logvar, 
            'mean_drift': [torch.zeros_like(post_mean)], 'logvar_drift': [torch.zeros_like(post_logvar)]}




def semantic_drift_esitimation(model, pre_model, dataloader, tokz, mem_std, args, device, pre_task):
    post_emb_all_curr = get_all_px_embeddings(model, dataloader, tokz, args, device)
    post_emb_all_pre = get_all_px_embeddings(pre_model, dataloader, tokz, args, device)

    post_mean_prev_data_pre_model = mem_std.memory[TASK2INFO[pre_task]['id']]['mean']
    post_logvar_prev_data_pre_model = mem_std.memory[TASK2INFO[pre_task]['id']]['logvar']
    if args.logvar_drift_flag :
        post_mean_prev_data_pre_model+=torch.sum(torch.stack(mem_std.memory[TASK2INFO[pre_task]['id']]['mean_drift']), dim=0)
        post_logvar_prev_data_pre_model+=torch.sum(torch.stack(mem_std.memory[TASK2INFO[pre_task]['id']]['logvar_drift']), dim=0)

    sample_level_semantic_drift_of_curr_task = post_emb_all_curr - post_emb_all_pre      
    distance = torch.norm(post_emb_all_pre - torch.unsqueeze(post_mean_prev_data_pre_model, 0), p=1, dim=1) 
    weight = torch.exp( - distance / (2 * args.sigma_drift * args.sigma_drift))
    weight_normed = weight/torch.sum(weight)
    mean_semantic_drift = torch.sum(sample_level_semantic_drift_of_curr_task * torch.unsqueeze(weight_normed, 1), dim=0) 
    
    logvar_semantic_drift = torch.zeros_like(mean_semantic_drift)
    if args.logvar_drift_flag:
        post_mean_curr_model = torch.mean(post_emb_all_curr, dim=0)
        post_mean_pre_model = torch.mean(post_emb_all_pre, dim=0)
        sample_level_logstd_drift_of_curr_task = torch.log(torch.abs(post_emb_all_curr - post_mean_curr_model) + 1e-8) - \
                                                torch.log(torch.abs(post_emb_all_pre - post_mean_pre_model)+1e-8) 

        distance = torch.norm(
                    (torch.log(torch.abs(post_emb_all_pre - post_mean_pre_model)+1e-8) - \
                                        0.5*torch.unsqueeze(post_logvar_prev_data_pre_model, 0)+1e-8), 
                p=1, dim=1) 

        weight = torch.exp( - distance / (2 * args.logvar_sigma_drift * args.logvar_sigma_drift))+1e-8
        weight_normed = weight/torch.sum(weight)
        logvar_semantic_drift = 2 * torch.sum(sample_level_logstd_drift_of_curr_task * torch.unsqueeze(weight_normed, 1), dim=0) 

    return mean_semantic_drift, logvar_semantic_drift


def get_all_px_embeddings(model, dataloader, tokz, args, device):
    with torch.no_grad():
        post_emb_list =[]
        model.eval()
        for i, data in enumerate(dataloader):
            vae_total = get_model_input(data, input_type='vae', step=100, tokz=tokz, args=args)
            p_mask, p_tokens, px_mask, px_tokens, \
                input_tokens, attention_mask, target_tokens, input_label_mask, task_id = vae_total
                    
            p_mask = p_mask.to(device)
            p_tokens = p_tokens.to(device)
            px_mask = px_mask.to(device)
            px_tokens = px_tokens.to(device) 

            post_out = model.encoder(input_ids=px_tokens, attention_mask=px_mask)  
            post_emb, _ = model.avg_attn(post_out[0], attention_mask=px_mask)
            post_emb_list.append(post_emb.cpu())
        model.train()
        post_emb_all = communicate_tensor(post_emb_list, pad_token=tokz.eos_token_id)
        return post_emb_all


def get_model_input(batch, input_type='vae', step=100, tokz=None, args=None):
    if input_type == 'vae':             
        if args.vae_train_input_type==0:
            px_tokens = batch['all_id']                                     
            px_mask = batch['all_mask'].contiguous()

            prompt_tokens = batch['prompt_id']                              
            prompt_mask = batch['prompt_mask'].contiguous()

            all_id = batch['all_id'][...,:-1].contiguous()  
            attn_mask = batch['all_mask'][..., :-1].contiguous()

            tgt_tokens = batch['all_id'][..., 1:].contiguous()              
            all_utter_label_mask = batch['all_utter_label_mask'][..., 1:].contiguous() 
            
            task_ids = batch['task_id']
            input_total = (px_mask, px_tokens, prompt_mask, prompt_tokens, 
                        all_id, attn_mask, tgt_tokens, all_utter_label_mask,
                        task_ids)
        else:
            pass

    else:                   
        all_tokens = batch['all_id'][..., :-1]                           
        all_mask = batch['all_mask'][..., :-1].contiguous()
        all_tgt_tokens = batch['all_id'][..., 1:].contiguous()           
        all_label_mask = batch['all_label_mask'][..., 1:].contiguous()
        input_total = (all_tokens, all_mask, all_label_mask, all_tgt_tokens,)

    return input_total 



def pad_tensor(tensor, length, pad_token=0):
    return torch.cat([tensor, tensor.new(tensor.size(0), length - tensor.size()[1]).fill_(pad_token)], dim=1)

def communicate_tensor(tensor_list, pad_token=0):

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




