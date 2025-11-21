import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

def split_prokaryotic_dataset(mat_path,output_path=None,  train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, random_state=37):

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

    print(f"\n数据划分成功，已保存到: {output_path}")

    return True

if __name__ == "__main__":
    split_prokaryotic_dataset(mat_path=r"C:\Users\lenovo\Desktop\TrashToTreasureWithDynamic\datasets\prokaryotic.mat")
