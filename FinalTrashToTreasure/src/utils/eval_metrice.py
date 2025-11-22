import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def eval_senti(results, truths, exclude_zero=False):


    results = results.cpu().numpy()
    truths = truths.cpu().numpy()

    if exclude_zero:
        mask = truths != 0
        results = results[mask]
        truths = truths[mask]
        print(f"After excluding class 0: {results.shape}, {truths.shape}")

    pred_labels = np.argmax(results, axis=1)

    metrics = {}

    metrics['accuracy'] = accuracy_score(truths, pred_labels)

    metrics['weighted_f1'] = f1_score(truths, pred_labels, average='weighted')

    metrics['macro_f1'] = f1_score(truths, pred_labels, average='macro')

    try:
        truths_onehot = np.eye(results.shape[1])[truths]
        metrics['auc'] = roc_auc_score(truths_onehot, results, multi_class='ovr')
    except Exception as e:
        print(f"AUC calculation error: {e}")
        metrics['auc'] = 0.0

    print(f"ACC: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    return metrics['accuracy']
