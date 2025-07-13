import torch
from settings import parse_args
import os
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, \
#                     AdamW, get_linear_schedule_with_warmup, Conv1D, BertTokenizer
from transformers import GPT2Tokenizer

from SnD_datasets import get_datasets

from SnD_model.utils import get_logger, seed_everything
from SnD_model.memory import Memory, Memory_std



os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_args()
logger = get_logger(args.log_file)

if args.local_rank in [0, -1]:
    logger.info('Pytorch Version: {}'.format(torch.__version__))
    for k, v in vars(args).items():
        logger.info("{}= {}".format(k, v))

seed_everything(args.seed)

cache_dir = os.path.join(args.output_dir, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)

dec_out = os.path.join(cache_dir,'dec_out')
tokz = GPT2Tokenizer.from_pretrained(args.gpt2_path)
tokz.save_pretrained(dec_out)

if args.local_rank in [0, -1]:
    logger.info('Loading datasets...'+'.'*10)

datasets = get_datasets(args.data_dir, args.tasks, tokz, 
                num_workers=args.num_workers, ctx_max_len=args.ctx_max_len, args=args)
for task in datasets.keys():
    logger.info('task:{}, max_q_len: {}, max_ans_length: {}, dataset_size: {}'.format(
        task, datasets[task]['train'].max_q_len, datasets[task]['train'].max_ans_len, len(datasets[task]['train'])))
logger.info('Finish loading datasets!')


mem_std= Memory_std()  if args.sta_flag else None
memory = Memory() if args.use_memory else None


if __name__ == '__main__':
    from SnD_model.trainer import Trainer
    trainer = Trainer(args, tokz, datasets, logger, cache_dir, memory=memory, memory_std=mem_std)
    trainer.train(args.tasks, continue_from_task=args.continue_from_task)

