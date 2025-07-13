import torch
import re
import json

from info import TASK2INFO


class PromptCLSDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, tokz, data_path, num_workers=8, ctx_max_len=100, args=None, max_ans_len=50):

        self.num_workers = num_workers
        self.data_path = data_path
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz

        self.max_ans_len = 0
        self.max_q_len = 0
        self.prompt_len = len(self.tokz.encode(self.apply_prompt(task_name, "", ctx_max_len)))

        self.pseudo_data_prompt = self._pseudo_data_prompt(task_name)   
        self.pseudo_ans_prompt = self._pseudo_ans_prompt()   
        self.pseudo_data_prompt_id = tokz.encode(self.pseudo_data_prompt) 
        self.pseudo_ans_prompt_id = tokz.encode(self.pseudo_ans_prompt) 

        self.pseudo_gene_prompt = self._pseudo_general_prompt() 
        self.pseudo_gene_prompt_id = tokz.encode(self.pseudo_gene_prompt) 


        with open(data_path, "r", encoding='utf-8') as f:
            if args.debug:
                data = [json.loads(i) for i in f.readlines()][:1024]
                data = [self.parse_example(i, args.ctx_max_ans_len) for i in data]   
            else:
                data = [json.loads(i) for i in f.readlines()]
                data = [self.parse_example(i, args.ctx_max_ans_len) for i in data]   
            
        self.ans_set = set()

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)
        
        print('ans set of task {}: {}'.format(task_name, self.ans_set ))

    @staticmethod
    def apply_prompt(task_name, text, ctx_max_len):  
        text = text.split(' ')[:ctx_max_len]
        return f"In the \"{task_name}\" task, which intent category best describes: \" {' '.join(text)} \"? Answer: " 
    
    @staticmethod
    def apply_general_prompt(text, ctx_max_len):
        text = text.split(' ')[:ctx_max_len]
        return f"In the current task, which intent category best describes: \" {' '.join(text)} \"? Answer: " 


    @staticmethod
    def _pseudo_data_prompt(task_name): 
        return f"In the \"{task_name}\" task, which intent category best describes: \""

    @staticmethod
    def _pseudo_general_prompt(): 
        return f"In the current task, which intent category best describes: \""
        
    @staticmethod
    def _pseudo_ans_prompt(): 
        return f" \"? Answer: "

    @staticmethod
    def parse_pseudo_data(text): 
        try:
            task_name = re.findall(r'In the "(.+?)" task, which intent category best describes:', text)[0]
            utter = re.findall(r'task, which intent category best describes: " (.+?) "\? Answer: ', text)[0]
            label = text.replace(f'In the "{task_name}" task, which intent category best describes: " {utter} "? Answer: ', '').strip()
            return {'task_name': task_name, 'utter': utter, 'label': label}
        except:
            return None

    @staticmethod
    def parse_example(example, ctx_max_ans_len):
        text = example['userInput']['text']
        ans = example['intent']  
        ans = ' '.join(ans.split(' ')[:ctx_max_ans_len]) 
        return text, ans 

    def parallel_tokenization(self, task_name, d, sta_flag=True):
        ori_utter = d[0]
        prompt = self._pseudo_data_prompt(task_name)  
        gene_prompt = self._pseudo_general_prompt()
        context = self.apply_prompt(task_name, d[0], self.ctx_max_len)
        general_context = self.apply_general_prompt(d[0], self.ctx_max_len) 
        ans = d[1].replace("-", " ").replace("_", " ") 
        if sta_flag: self.ans_set.add(ans.lower().strip()) 

        prompt_id = self.tokz.encode(prompt)
        gene_prompt_id = self.tokz.encode(gene_prompt)
        input_id = self.tokz.encode(d[0])
        context_id = self.tokz.encode(context)
        general_context_id = self.tokz.encode(general_context)
        ans_id = self.tokz.encode(ans)
        utter_id = self.tokz.encode(" "+ori_utter) 

        if sta_flag: 
            self.max_ans_len = max(self.max_ans_len, len(ans_id) + 1)
            self.max_q_len = max(self.max_q_len, len(utter_id) + 1)


        return {
            'utter_id': utter_id + [self.tokz.eos_token_id],
            'posterior_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id],
            'input_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id],
            'prompt_id': [self.tokz.bos_token_id] + prompt_id ,
            'gene_prompt_id': [self.tokz.bos_token_id] + gene_prompt_id + [self.tokz.eos_token_id],
            'gene_posterior_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], 
            'gene_input_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], 
            'all_id': [self.tokz.bos_token_id] + context_id + ans_id + [self.tokz.eos_token_id], 
            'context_id': [self.tokz.bos_token_id] + context_id,
            'general_context_id': [self.tokz.bos_token_id] + general_context_id,
            'gene_all_id': [self.tokz.bos_token_id] + general_context_id + ans_id + [self.tokz.eos_token_id], 
            'ans_id': ans_id + [self.tokz.eos_token_id],
            'task_id': TASK2INFO[task_name]['id'],
            }

    def data_tokenization(self, task_name, data, sta_flag=True):

        data = [self.parallel_tokenization(task_name, i, sta_flag=sta_flag) for i in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def pseudo_data_tokenization(cls, task_name, data, ctx_max_len=100, tokz=None):

        def pseudo_parallel_tokenization(task_name, d, ctx_max_len=100, tokz=None):

            ori_utter = d[0]
            prompt = cls._pseudo_data_prompt(task_name)  
            gene_prompt = cls._pseudo_general_prompt()
            context = cls.apply_prompt(task_name, d[0], ctx_max_len) 
            general_context = cls.apply_general_prompt(d[0], ctx_max_len) 

            ans = d[1].replace("-", " ").replace("_", " ")  

            prompt_id = tokz.encode(prompt)
            gene_prompt_id = tokz.encode(gene_prompt)
            input_id = tokz.encode(d[0])
            context_id = tokz.encode(context)
            general_context_id = tokz.encode(general_context)
            ans_id = tokz.encode(ans)
            utter_id = tokz.encode(" "+ori_utter)  

            return {
                'utter_id': utter_id + [tokz.eos_token_id],
                'posterior_id': [tokz.bos_token_id] + prompt_id + utter_id + [tokz.eos_token_id], 
                'input_id': [tokz.bos_token_id] + prompt_id + utter_id + [tokz.eos_token_id], 
                'prompt_id': [tokz.bos_token_id] + prompt_id , 
                'gene_prompt_id': [tokz.bos_token_id] + gene_prompt_id + [tokz.eos_token_id],
                'gene_posterior_id': [tokz.bos_token_id] + gene_prompt_id + utter_id + [tokz.eos_token_id], 
                'gene_input_id': [tokz.bos_token_id] + gene_prompt_id + utter_id + [tokz.eos_token_id], 
                'all_id': [tokz.bos_token_id] + context_id + ans_id + [tokz.eos_token_id],
                'context_id': [tokz.bos_token_id] + context_id,
                'general_context_id': [tokz.bos_token_id] + general_context_id,
                'gene_all_id': [tokz.bos_token_id] + general_context_id + ans_id + [tokz.eos_token_id],
                'ans_id': ans_id + [tokz.eos_token_id], 
                'task_id': TASK2INFO[task_name]['id'],
                }

        data = [pseudo_parallel_tokenization(task_name, i, ctx_max_len, tokz) for i in data]
        return data





class MixedCLSDataset(PromptCLSDataset):
    def __init__(self, task2data,  tokz, ctx_max_len=100, curr_data=None):
        '''
        task2data : {'task_name': [data1, data2, data3, ...]}
        '''
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        
        self.max_ans_len = 0
        self.max_q_len = 0

        self.ans_set = set()

        self.data = []
        for task in task2data:
            self.task_name = task
            self.task_id = TASK2INFO[task]['id']
            self.data += self.data_tokenization(task, task2data[task])
        if curr_data is not None:
            self.data += curr_data

class PseudoCLSDataset(PromptCLSDataset):
    def __init__(self, taskname, data, tokz, ctx_max_len=100):
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0
        self.data = self.data_tokenization(taskname, data)