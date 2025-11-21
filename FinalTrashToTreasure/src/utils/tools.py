import os
import numpy as np
import torch

def save_model(args, model, name=''):

    name = 'best_model'
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def _get_class_weights(train_loader,device):

    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    labels = all_labels
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    total_samples = len(labels)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return class_weights