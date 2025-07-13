import string
from collections import OrderedDict

from sacrebleu import corpus_bleu
from dictdiffer import diff
import numpy as np

from info import TASK2INFO




def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return lower(s)


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))



def compute_em(predictions, references, xlingual=False):
    exact_match = 0
    for pred, gold in zip(predictions, references):
        exact_match += exact_match_score(pred, gold, xlingual=False)
    return np.round(100.0 * exact_match / len(references), 3)






def compute_metrics(predictions, references, xlingual=False, task_name=None):

    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    metrics = OrderedDict()
    for eval_metric in TASK2INFO[task_name]['eval_metric']:
        if eval_metric == 'em':
            metrics['em'] =  compute_em(predictions, references, xlingual=xlingual)
        if eval_metric == 'bleu':
            pass
    return metrics

