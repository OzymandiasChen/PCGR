
import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_vae_loss(device, model, loss_fn, beta, vae_total, distill=False, prev_model=None, 
                        args=None, mem_std=None):

    if args.vae_train_input_type==0:
        px_mask, px_tokens, prompt_mask, prompt_tokens, \
            input_tokens, attention_mask, target_tokens, input_label_mask, task_ids = vae_total
        input_tokens = input_tokens.to(device)          
        attention_mask = attention_mask.to(device)
        target_tokens = target_tokens.to(device)        

        px_mask = px_mask.to(device)                    
        px_tokens = px_tokens.to(device)
        prompt_mask = prompt_mask.to(device)
        prompt_tokens = prompt_tokens.to(device) 
        task_ids = task_ids.to(device)

        outputs = model(input_ids=input_tokens, attention_mask=attention_mask, 
                    prompt_tokens=prompt_tokens, prompt_mask=prompt_mask, 
                    px_mask=px_mask, px_tokens=px_tokens,
                    task_ids = task_ids, mem_std=mem_std, phase='train')   
    else:
        pass



    logits = outputs[0] 
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    if attention_mask is not None:
        attention_mask = attention_mask.type(torch.bool)
        attention_mask = attention_mask.to(device)
        loss_mask = (input_label_mask>0).view(-1)

    if distill:     
        assert prev_model is not None
        prev_model.eval()
        with torch.no_grad():
            if args.vae_type in ['prototype_cvae']:
                if args.vae_train_input_type == 0:
                    teacher_outputs = prev_model(input_ids=input_tokens, attention_mask=attention_mask, 
                        prompt_tokens=prompt_tokens, prompt_mask=prompt_mask, 
                        px_mask=px_mask, px_tokens=px_tokens,
                        task_ids = task_ids, mem_std=mem_std, phase='train')              
                    teacher_logits = teacher_outputs[0]
                    teacher_logits = teacher_logits.contiguous()
        ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1), 
                                teacher_logits.view(-1, teacher_logits.size(-1)))    
    else:
        ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)) 
 
    ce_loss_masked = ce_loss.where(loss_mask.cuda(), torch.tensor(0.0).cuda())
    ce_loss = ce_loss_masked.sum() / loss_mask.sum() 

    kl_loss = kl_loss.mean()
    loss = ce_loss + beta * kl_loss * args.kl_loss_weight
    return loss, ce_loss, kl_loss


def compute_lm_loss(device, model, loss_fn, lm_total,  distill=False, prev_model=None, args=None, mem_std=None):
    lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens = lm_total
    lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens = \
        lm_input_tokens.to(device), lm_attention_mask.to(device), \
        lm_input_label_mask.to(device),lm_target_tokens.to(device)

    model=model.to(device)
    model.decoder.train()
    outputs = model.decoder(input_ids=lm_input_tokens)
    logits = outputs.logits.contiguous()
    
    loss_mask = (lm_input_label_mask>0).view(-1)
    loss_padding_mask = (lm_attention_mask>0).view(-1)
    logits = logits.view(-1, logits.size(-1))
    if distill:
        assert prev_model is not None
        prev_model.eval()
        with torch.no_grad():
            tc_outputs = prev_model.decoder(input_ids=lm_input_tokens)
            teacher_logits = tc_outputs.logits.contiguous()
        ce_loss = loss_fn(logits, lm_target_tokens.view(-1), 
                            teacher_logits.view(-1, teacher_logits.size(-1)))
    else:
        ce_loss = loss_fn(logits, lm_target_tokens.view(-1))

    lm_loss_masked = ce_loss.where(loss_mask.cuda(), torch.tensor(0.0).cuda())
    lm_pad_loss_masked = ce_loss.where(loss_padding_mask.cuda(), torch.tensor(0.0).cuda())
    qa_loss = lm_loss_masked.sum() / loss_mask.sum()                        
    lm_loss = lm_pad_loss_masked.sum() / loss_padding_mask.sum()            

    loss = qa_loss + 0.5 * lm_loss

    return loss


class KDLoss(nn.Module):
    def __init__(self, KD_term=0.0, T=1.0):
        super(KDLoss, self).__init__()
        assert 0 <= KD_term <=1
        assert 0 < T
        self.KD_term = KD_term
        self.T = T
    
    def forward(self, output_logits, targets, teacher_logits=None):
        if teacher_logits is None:
            return F.cross_entropy(output_logits, targets, reduction='none')
        else:  
            KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
                F.softmax(teacher_logits / self.T, dim=1), reduction='none')
            KD_loss = torch.sum(KD_loss, dim=1)
            CE_loss = F.cross_entropy(output_logits, targets, reduction='none')
            return KD_loss * self.KD_term * self.T * self.T + CE_loss * (1 - self.KD_term)   


def kl_divergence(mean1, logvar1, mean2, logvar2):
    exponential = logvar1 - logvar2 - \
        torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
        torch.exp(logvar1 - logvar2) + 1
    result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    return result.mean()  

