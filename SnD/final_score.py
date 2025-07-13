import json
import os, sys
import numpy as np
from settings import parse_args
from collections import OrderedDict

from info import TASK2INFO

args = parse_args()
last_task, task_num = args.tasks[-1], len(args.tasks)



test_res = open(os.path.join(args.res_dir, 'res.txt'), 'w')
score_all_time=[]
with open(os.path.join(args.output_dir, "metrics.json"),"r") as f:
    score_all_time = [json.loads(row) for row in f.readlines()]

last_dict_by_keys = {}
for d in score_all_time:
    num_keys = len(d)
    last_dict_by_keys[num_keys] = d




model_eval_rst = OrderedDict()
for idx, task_train in enumerate(args.tasks):
    model_eval_rst[task_train] = last_dict_by_keys[idx+1]



print('************ overall performance  ***************')
overall = OrderedDict()


AP = 0
for task_eval in args.tasks:
    m = TASK2INFO[task_eval]['metric']
    P = model_eval_rst[last_task][task_eval][m]
    AP += P 
overall['AP'] = np.round(AP / task_num, 3)


for k, v in overall.items():
    print("model-[%10s] : %.3f" % (k, v)) 

