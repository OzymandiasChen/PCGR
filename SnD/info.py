


ORDER_DICT = {   
    'order_0': ['example_task_0', 'example_rask_1']
}


TASK2INFO = {   
    'example_task_0': {   
            'dataset_class': 'PromptCLSDataset',
            'dataset_folder': 'example_task_0',
            'eval_metric': {'em'},
            'id': 0,
            'metric': 'em',
            'part_prompt': 'which intent category best describes:',
            'task_type': 'intent'
        },
    'example_task_1': {   
            'dataset_class': 'PromptCLSDataset',
            'dataset_folder': 'example_task_1',
            'eval_metric': {'em'},
            'id': 1,
            'metric': 'em',
            'part_prompt': 'which intent category best describes:',
            'task_type': 'intent'
        }
}

IDTOTASK = {   
    0: 'example_task_0',
    1: 'example_task_1',
}


TASKTOID = {   
    'example_task_0': 0,
    'example_task_1': 1
}



