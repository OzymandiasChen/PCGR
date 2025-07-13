
import os
import torch


from info import TASK2INFO

from SnD_datasets.intent_dataset import PromptCLSDataset, MixedCLSDataset, PseudoCLSDataset
from SnD_datasets.pad_batch import PadBatchSeq
from SnD_datasets.slot_dataset import PromptSlotTaggingDataset, MixedSlotTaggingDataset
from SnD_datasets.utilis import pad_seq



from SnD_datasets.summary_dataset import SUMMARYDataset, SUMMARYDataset_1
from SnD_datasets.simplification_dataset import SIMPLIFICATIONDataset 



def get_datasets(path, tasks, tokz, num_workers=8, ctx_max_len=100, args=None):
    res = {}

    for task in tasks:
        res[task] = {}
        info = TASK2INFO[task]
        res[task]['train'] = eval(TASK2INFO[task]['dataset_class'])(
            task, tokz, 
            os.path.join(path, info['dataset_folder'], '', 'train.json'), 
            num_workers=num_workers, ctx_max_len=ctx_max_len, args=args)
        res[task]['val'] = eval(TASK2INFO[task]['dataset_class'])(
            task, tokz, 
            os.path.join(path, info['dataset_folder'], '','valid.json'), 
            num_workers=num_workers, ctx_max_len=ctx_max_len, args=args)
        res[task]['test'] = eval(TASK2INFO[task]['dataset_class'])(
            task, tokz, 
            os.path.join(path, info['dataset_folder'], '','test.json'), 
            num_workers=num_workers, ctx_max_len=ctx_max_len, args=args)

    return res



class MixDataset(torch.utils.data.Dataset):

    def __init__(self, task2data,  tokz, ctx_max_len=100, curr_data=None):

        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        
        self.max_ans_len = 0 
        self.max_q_len = 0

        self.data = []
        for task in task2data.keys():
            self.data += eval(TASK2INFO[task]['dataset_class']).pseudo_data_tokenization(
                            task, task2data[task], ctx_max_len=ctx_max_len, tokz=tokz)
        if curr_data is not None:
            self.data += curr_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

