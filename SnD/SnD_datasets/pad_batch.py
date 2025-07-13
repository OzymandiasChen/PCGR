

import torch


from SnD_datasets.utilis import pad_seq





class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self

    def __repr__(self):
        return self.data.__repr__()
    
    def __str__(self):
        return self.data.__str__()
    
    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()





class PadBatchSeq:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        utter_id = [i['utter_id'] for i in batch]
        input_id = [i['input_id'] for i in batch]
        posterior_id = [i['posterior_id'] for i in batch]
        prompt_id = [i['prompt_id'] for i in batch]
        gene_prompt_id = [i['gene_prompt_id'] for i in batch]
        gene_input_id = [i['gene_input_id'] for i in batch]
        gene_posterior_id = [i['gene_posterior_id'] for i in batch]
        context_id = [i['context_id'] for i in batch]
        general_context_id = [i['general_context_id'] for i in batch]
        ans_id = [i['ans_id'] for i in batch]
        all_id = [i['all_id'] for i in batch]
        gene_all_id = [i['gene_all_id'] for i in batch]
        task_id = [i['task_id'] for i in batch]


        all_lens = [len(i) for i in all_id]
        gene_all_lens = [len(i) for i in gene_all_id]
        context_lens = [len(i) for i in context_id]
        general_context_lens = [len(i) for i in general_context_id]
        ans_lens = [len(i) for i in ans_id]
        prompt_lens = [len(i) for i in prompt_id]
        gene_prompt_lens = [len(i) for i in gene_prompt_id]
        input_lens = [len(i) for i in input_id]
        gene_input_lens = [len(i) for i in gene_input_id]
        posterior_lens = [len(i) for i in posterior_id]
        gene_posterior_lens = [len(i) for i in gene_posterior_id]
        utter_lens = [len(i) for i in utter_id]

        ans_mask = torch.ByteTensor([[1] * ans_lens[i] + [0] * (max(ans_lens)-ans_lens[i]) for i in range(len(ans_id))]) 
        context_mask = torch.ByteTensor([[1] * context_lens[i] + [0] * (max(context_lens)-context_lens[i]) for i in range(len(context_id))]) 
        general_context_mask = torch.ByteTensor([[1] * general_context_lens[i] + [0] * (max(general_context_lens)-general_context_lens[i]) for i in range(len(general_context_id))]) 
        prompt_mask = torch.ByteTensor([[1] * prompt_lens[i] + [0] * (max(prompt_lens)-prompt_lens[i]) for i in range(len(prompt_id))]) 
        gene_prompt_mask = torch.ByteTensor([[1] * gene_prompt_lens[i] + [0] * (max(gene_prompt_lens)-gene_prompt_lens[i]) for i in range(len(gene_prompt_id))]) 
        input_mask = torch.ByteTensor([[1] * input_lens[i] + [0] * (max(input_lens)-input_lens[i]) for i in range(len(input_id))]) 
        gene_input_mask = torch.ByteTensor([[1] * gene_input_lens[i] + [0] * (max(gene_input_lens)-gene_input_lens[i]) for i in range(len(gene_input_id))]) 
        utter_mask = torch.ByteTensor([[1] * utter_lens[i] + [0] * (max(utter_lens)-utter_lens[i]) for i in range(len(utter_id))]) 
        posterior_mask = torch.ByteTensor([[1] * posterior_lens[i] + [0] * (max(posterior_lens)-posterior_lens[i]) for i in range(len(posterior_id))]) 
        gene_posterior_mask = torch.ByteTensor([[1] * gene_posterior_lens[i] + [0] * (max(gene_posterior_lens)-gene_posterior_lens[i]) for i in range(len(gene_posterior_id))]) 
        all_mask = torch.ByteTensor([[1] * all_lens[i] + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))]) 
        gene_all_mask = torch.ByteTensor([[1] * gene_all_lens[i] + [0] * (max(gene_all_lens)-gene_all_lens[i]) for i in range(len(gene_all_id))]) 


        all_label_mask = torch.ByteTensor([[0] * (context_lens[i]) + [1] * (all_lens[i] - context_lens[i]) + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))]) 
        gene_all_label_mask = torch.ByteTensor([[0] * (general_context_lens[i]) + [1] * (gene_all_lens[i] - general_context_lens[i]) + [0] * (max(gene_all_lens)-gene_all_lens[i]) for i in range(len(gene_all_id))]) 
        input_label_mask = torch.ByteTensor([[0] * (prompt_lens[i]-1) +[1] * (input_lens[i]-prompt_lens[i]+1) + [0] * (max(input_lens)-input_lens[i]) for i in range(len(input_id))])
        gene_input_label_mask = torch.ByteTensor([[0] * (gene_prompt_lens[i]-1) +[1] * (gene_input_lens[i]-gene_prompt_lens[i]+1) + [0] * (max(gene_input_lens)-gene_input_lens[i]) for i in range(len(gene_input_id))]) 

        all_utter_label_mask = torch.ByteTensor([[0] * (prompt_lens[i]-1) + [1] * (all_lens[i] - (prompt_lens[i]-1)) + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))]) 


        res = {}
        res['prompt_id'] = torch.tensor([pad_seq(i, self.pad_id, max(prompt_lens)) for i in prompt_id], dtype=torch.long)
        res['gene_prompt_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_prompt_lens)) for i in gene_prompt_id], dtype=torch.long)
        res['input_id'] = torch.tensor([pad_seq(i, self.pad_id, max(input_lens)) for i in input_id], dtype=torch.long)
        res['gene_input_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_input_lens)) for i in gene_input_id], dtype=torch.long)
        res['posterior_id'] = torch.tensor([pad_seq(i, self.pad_id, max(posterior_lens)) for i in posterior_id], dtype=torch.long)
        res['gene_posterior_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_posterior_lens)) for i in gene_posterior_id], dtype=torch.long)
        res["ans_id"] = torch.tensor([pad_seq(i, self.pad_id, max(ans_lens)) for i in ans_id], dtype=torch.long)      
        res["context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(context_lens), pad_left=True) for i in context_id], dtype=torch.long)      
        res["general_context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(general_context_lens), pad_left=True) for i in general_context_id], dtype=torch.long)      
        res["all_id"] = torch.tensor([pad_seq(i, self.pad_id, max(all_lens)) for i in all_id], dtype=torch.long)      
        res["gene_all_id"] = torch.tensor([pad_seq(i, self.pad_id, max(gene_all_lens)) for i in gene_all_id], dtype=torch.long)      
        res["utter_id"] = torch.tensor([pad_seq(i, self.pad_id, max(utter_lens)) for i in utter_id], dtype=torch.long)

        res["all_lens"] = torch.tensor(all_lens, dtype=torch.long)   
        res["context_lens"] = torch.tensor(context_lens, dtype=torch.long)   
        res["general_context_lens"] = torch.tensor(general_context_lens, dtype=torch.long)   
        res["ans_lens"] = torch.tensor(ans_lens, dtype=torch.long)   
        res["prompt_lens"] = torch.tensor(prompt_lens, dtype=torch.long) 
       
        res["all_mask"], res["context_mask"], res['prompt_mask'], res['ans_mask'], res['input_mask'] = all_mask, context_mask, prompt_mask, ans_mask, input_mask
        res['utter_mask'] = utter_mask
        res['posterior_mask'] = posterior_mask
        res['input_label_mask'] = input_label_mask 
        res['all_label_mask'] =  all_label_mask
        res['all_utter_label_mask'] = all_utter_label_mask 

        res['general_context_mask'] = general_context_mask
        res['gene_all_mask'] = gene_all_mask
        res['gene_input_mask'] = gene_input_mask
        res['gene_input_label_mask'] = gene_input_label_mask 
        res['gene_prompt_mask'] = gene_prompt_mask
        res['gene_all_label_mask'] =  gene_all_label_mask
        res['gene_input_label_mask'] = gene_input_label_mask 
        res['gene_posterior_mask'] = gene_posterior_mask
        res['task_id'] = torch.tensor(task_id, dtype=torch.long)  
        

        return PinnedBatch(res)


