import random

import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

def split_handwritten_dataset(mat_path, output_path, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=37):

    mat_content = sio.loadmat(mat_path)

    X = mat_content['X']
    feas = mat_content['feas']
    truth = mat_content['truth']

    if truth.ndim > 1:
        truth = truth.flatten()

    n_samples = len(truth)

    views_data = []
    n_views = X.shape[1]

    for i in range(n_views):
        view_data = X[0, i]

        if view_data.ndim == 1:
            view_data = view_data.reshape(-1, 1)

        views_data.append(view_data)


    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)
    n_test = n_samples - n_train - n_valid


    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]


    save_dict = {}


    save_dict['X_original'] = X
    save_dict['feas_original'] = feas
    save_dict['truth_original'] = truth

    for i, view_data in enumerate(views_data):
        view_name = f'view{i + 1}'

        save_dict[f'{view_name}_train'] = view_data[train_indices]
        save_dict[f'{view_name}_valid'] = view_data[valid_indices]
        save_dict[f'{view_name}_test'] = view_data[test_indices]

    save_dict['truth_train'] = truth[train_indices]
    save_dict['truth_valid'] = truth[valid_indices]
    save_dict['truth_test'] = truth[test_indices]

    save_dict['split_info'] = np.array(
        [f'train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}, seed:{random_state}'])

    sio.savemat(output_path, save_dict)


    print(f"Data split successfully, saved to: {output_path}")

    return True


def split_landuse_dataset(mat_path, output_path, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=37):


    mat_content = sio.loadmat(mat_path)


    X = mat_content['X']
    Y = mat_content['Y']

    if Y.ndim > 1:
        Y = Y.flatten()

    n_samples=len(Y)

    views_data = []
    n_views = X.shape[1]

    for i in range(n_views):
        view_data = X[0, i]

        # Ensure data is 2D array
        if view_data.ndim == 1:
            view_data = view_data.reshape(-1, 1)

        views_data.append(view_data)



    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)


    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)
    n_test = n_samples - n_train - n_valid



    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    save_dict = {}


    save_dict['X_original'] = X
    save_dict['Y_original'] = Y


    for i, view_data in enumerate(views_data):
        view_name = f'view{i + 1}' if len(views_data) > 1 else 'X'

        save_dict[f'{view_name}_train'] = view_data[train_indices]
        save_dict[f'{view_name}_valid'] = view_data[valid_indices]
        save_dict[f'{view_name}_test'] = view_data[test_indices]


    save_dict['truth_train'] = Y[train_indices]
    save_dict['truth_valid'] = Y[valid_indices]
    save_dict['truth_test'] = Y[test_indices]


    save_dict['split_info'] = np.array(
        [f'train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}, seed:{random_state}'])



    sio.savemat(output_path, save_dict)

    print(f"Data split successfully, saved to: {output_path}")

    return True


def split_scene15_dataset(mat_path, output_path, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=37):

    mat_content = sio.loadmat(mat_path)

    X = mat_content['X']
    Y = mat_content['Y']

    if Y.ndim > 1:
        Y = Y.flatten()

    n_samples = len(Y)
    n_views = X.shape[1]

    views_data = []
    for i in range(n_views):
        view_data = X[0, i]

        # Ensure data is 2D array
        if view_data.ndim == 1:
            view_data = view_data.reshape(-1, 1)

        views_data.append(view_data)

    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)
    n_test = n_samples - n_train - n_valid

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    save_dict = {}

    save_dict['X_original'] = X
    save_dict['Y_original'] = Y

    for i, view_data in enumerate(views_data):
        view_name = f'view{i + 1}' if len(views_data) > 1 else 'X'

        save_dict[f'{view_name}_train'] = view_data[train_indices]

        save_dict[f'{view_name}_valid'] = view_data[valid_indices]

        save_dict[f'{view_name}_test'] = view_data[test_indices]


    save_dict['truth_train'] = Y[train_indices]
    save_dict['truth_valid'] = Y[valid_indices]
    save_dict['truth_test'] = Y[test_indices]


    save_dict['split_info'] = np.array(
        [f'train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}, seed:{random_state}'])

    sio.savemat(output_path, save_dict)

    print(f"Data split successfully, saved to: {output_path}")

    return True

def split_Reuters_dataset(mat_path, output_path, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=42):
    mat_content = sio.loadmat(mat_path)

    X = mat_content['X']
    Y = mat_content['Y']

    if Y.ndim > 1:
        Y = Y.flatten()

    n_samples = len(Y)

    views_data = []
    n_views = X.shape[1]

    for i in range(n_views):
        view_data = X[0, i]

        if view_data.ndim == 1:
            view_data = view_data.reshape(-1, 1)

        views_data.append(view_data)
        print(f"  View {i + 1} dimension: {view_data.shape}")

    # Verify that ratio sum equals 1
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratio sum should be 1.0, current is {total_ratio}")

    train_indices, temp_indices = train_test_split(
        np.arange(n_samples),
        train_size=train_ratio,
        stratify=Y,
        random_state=random_state
    )

    valid_ratio_remaining = valid_ratio / (valid_ratio + test_ratio)

    valid_indices, test_indices = train_test_split(
        temp_indices,
        train_size=valid_ratio_remaining,
        stratify=Y[temp_indices],
        random_state=random_state
    )

    def check_stratification(y, indices, set_name):
        unique, counts = np.unique(y[indices], return_counts=True)
        total = len(indices)
        proportions = counts / total
        original_proportions = np.array([np.sum(y == cls) for cls in unique]) / len(y)

        return proportions

    train_props = check_stratification(Y, train_indices, "train set")
    valid_props = check_stratification(Y, valid_indices, "validation set")
    test_props = check_stratification(Y, test_indices, "test set")

    save_dict = {}

    save_dict['X_original'] = X
    save_dict['Y_original'] = Y

    for key in ['feanames', 'lenSmp']:
        if key in mat_content:
            save_dict[f'{key}_original'] = mat_content[key]

    for i, view_data in enumerate(views_data):
        view_name = f'view{i + 1}'

        save_dict[f'{view_name}_train'] = view_data[train_indices]
        save_dict[f'{view_name}_valid'] = view_data[valid_indices]
        save_dict[f'{view_name}_test'] = view_data[test_indices]

    save_dict['truth_train'] = Y[train_indices]
    save_dict['truth_valid'] = Y[valid_indices]
    save_dict['truth_test'] = Y[test_indices]

    save_dict['split_info'] = np.array([
        f'train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}',
        f'random_state:{random_state}',
        f'n_train:{len(train_indices)}, n_valid:{len(valid_indices)}, n_test:{len(test_indices)}',
        f'stratified:True'
    ])

    save_dict['train_indices'] = train_indices
    save_dict['valid_indices'] = valid_indices
    save_dict['test_indices'] = test_indices

    unique_classes, class_counts = np.unique(Y, return_counts=True)
    class_proportions = class_counts / n_samples
    save_dict['class_distribution'] = np.array([unique_classes, class_counts, class_proportions])

    view_dims = [view_data.shape[1] for view_data in views_data]
    save_dict['view_dimensions'] = np.array(view_dims)
    save_dict['view_names'] = np.array(['English', 'French', 'German', 'Italian', 'Spanish'][:n_views])

    sio.savemat(output_path, save_dict)

    print(f"\nData split successfully, saved to: {output_path}")

    return True

def split_prokaryotic_dataset(mat_path ,output_path,  train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=42):

    mat_content = sio.loadmat(mat_path)

    truth = mat_content['truth']
    text = mat_content['text']
    proteome_comp = mat_content['proteome_comp']
    gene_repert = mat_content['gene_repert']

    if truth.ndim > 1:
        truth = truth.flatten()

    n_samples = len(truth)

    views = {
        'view1': text,
        'view2': proteome_comp,
        'view3': gene_repert
    }

    train_indices, temp_indices = train_test_split(
        np.arange(n_samples),
        train_size=train_ratio,
        stratify=truth,
        random_state=random_state
    )

    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)

    valid_indices, test_indices = train_test_split(
        temp_indices,
        train_size=valid_ratio_adjusted,
        stratify=truth[temp_indices],
        random_state=random_state
    )

    save_dict = {}

    save_dict['truth_original'] = truth
    save_dict['text_original'] = text
    save_dict['proteome_comp_original'] = proteome_comp
    save_dict['gene_repert_original'] = gene_repert

    save_dict['train_indices'] = train_indices
    save_dict['valid_indices'] = valid_indices
    save_dict['test_indices'] = test_indices

    for view_name, view_data in views.items():
        save_dict[f'{view_name}_train'] = view_data[train_indices]
        save_dict[f'{view_name}_valid'] = view_data[valid_indices]
        save_dict[f'{view_name}_test'] = view_data[test_indices]

    save_dict['truth_train'] = truth[train_indices]
    save_dict['truth_valid'] = truth[valid_indices]
    save_dict['truth_test'] = truth[test_indices]

    sio.savemat(output_path, save_dict)

    print(f"Data split successfully, saved to: {output_path}")

    return True

def split_3sources_dataset(mat_path, output_path, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=37):


    mat_content = sio.loadmat(mat_path)

    truth = mat_content['truth']
    bbc = mat_content['bbc']
    guardian = mat_content['guardian']
    reuters = mat_content['reuters']

    if truth.ndim > 1:
        truth = truth.flatten()

    n_samples = len(truth)

    views = {
        'view1': bbc,
        'view2': guardian,
        'view3': reuters
    }

    train_indices, temp_indices = train_test_split(
        np.arange(n_samples),
        train_size=train_ratio,
        stratify=truth,
        random_state=random_state
    )

    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)

    valid_indices, test_indices = train_test_split(
        temp_indices,
        train_size=valid_ratio_adjusted,
        stratify=truth[temp_indices],
        random_state=random_state
    )

    save_dict = {}

    save_dict['truth_original'] = truth
    save_dict['bbc_original'] = bbc
    save_dict['guardian_original'] = guardian
    save_dict['reuters_original'] = reuters

    save_dict['train_indices'] = train_indices
    save_dict['valid_indices'] = valid_indices
    save_dict['test_indices'] = test_indices

    for view_name, view_data in views.items():
        save_dict[f'{view_name}_train'] = view_data[train_indices]
        save_dict[f'{view_name}_valid'] = view_data[valid_indices]
        save_dict[f'{view_name}_test'] = view_data[test_indices]

    save_dict['truth_train'] = truth[train_indices]
    save_dict['truth_valid'] = truth[valid_indices]
    save_dict['truth_test'] = truth[test_indices]


    sio.savemat(output_path, save_dict)

    print(f"Data split successfully, saved to: {output_path}")

    return True