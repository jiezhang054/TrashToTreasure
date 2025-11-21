import scipy.io as sio
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

import split

class MMDataset(Dataset):
    def __init__(self, config,mode='train'):
        self.config = config
        self.mode = mode
        self.num_views = self.config.num_views
        DATA_MAP = {
            'Synthetic': self.__init_Synthetic,
            '3Sources': self.__init_3Sources,
            'Handwritten': self.__init_Handwirtten,
            'LandUse21': self.__init_LandUse21,
            'Prokaryotic': self.__init_Prokaryotic,
            'Scene15': self.__init_Scene15,
            'Reuters':self.__init_Reuters,
        }

        # 根据配置选择数据集初始化方法
        if config.dataset in DATA_MAP:
            DATA_MAP[config.dataset]()
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

    def check_split(self):
        original_path = Path(self.config.dataset_dir)
        stem = original_path.stem  # 获取文件名（不含后缀）
        suffix = original_path.suffix  # 获取文件后缀
        split_path = original_path.parent / f"{stem}_622{suffix}"

        if not split_path.exists():
            print(f"划分后数据集不存在，需要划分: {split_path}")
            return False,split_path
        else:
            return True,split_path

    def convert_sparse_to_dense(self,data):
        if hasattr(data, 'toarray'):
            return data.toarray()
        return data

    def normalize_data(self,data_list):
        """归一化每个视图的数据"""
        normalized_list = []
        for data in data_list:

            mean = data.mean(dim=0, keepdim=True)
            std = data.std(dim=0, keepdim=True) + 1e-8


            normalized = (data - mean) / std
            normalized_list.append(normalized)

        return normalized_list

    def get_data_and_labels(self,num_views,mat_content):
        data = []
        if self.mode == 'train':
            for i in range(num_views):
                view_data = mat_content[f'view{i + 1}_train']
                view_data = self.convert_sparse_to_dense(view_data)
                view_data = torch.FloatTensor(view_data)
                data.append(view_data)
            labels = mat_content['truth_train'].flatten()
        elif self.mode == 'valid':
            for i in range(num_views):
                view_data = mat_content[f'view{i + 1}_valid']
                view_data = self.convert_sparse_to_dense(view_data)
                view_data = torch.FloatTensor(view_data)
                data.append(view_data)
            labels = mat_content['truth_valid'].flatten()
        elif self.mode == 'test':
            for i in range(num_views):
                view_data = mat_content[f'view{i + 1}_test']
                view_data = self.convert_sparse_to_dense(view_data)
                view_data = torch.FloatTensor(view_data)
                data.append(view_data)
            labels = mat_content['truth_test'].flatten()
        else:
            raise ValueError("mode必须是 'train', 'valid' 或 'test'")

        if labels.ndim > 1:
            labels = self.labels.squeeze()
        if min(labels) == 1:
            labels = labels-1

        labels = torch.LongTensor(labels)

        return data,labels


    def __init_Synthetic(self):
        mat_content = sio.loadmat(self.config.dataset_dir)
        self.data = []
        if self.mode == 'train':
            self.data.append(torch.FloatTensor(mat_content['X1_train']))
            self.data.append(torch.FloatTensor(mat_content['X2_train']))
            self.labels = mat_content['gt_train']
        else:

            self.data.append(torch.FloatTensor(mat_content['X1_test']))
            self.data.append(torch.FloatTensor(mat_content['X2_test']))
            self.labels = mat_content['gt_test']

        if self.labels.ndim > 1:
            self.labels = self.labels.squeeze()
        if min(self.labels) == 1:
            self.labels = self.labels-1
        self.labels = torch.LongTensor(self.labels)
        self.data = self.normalize_data(self.data)

    def __init_3Sources(self):
        split_flag,split_path = self.check_split()
        if not split_flag:
            split.split_3sources_dataset(mat_path=self.config.dataset_dir,output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)


    def __init_Handwirtten(self):

        split_flag, split_path = self.check_split()
        if not split_flag:
            split.split_handwritten_dataset(mat_path=self.config.dataset_dir, output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)
        self.data = self.normalize_data(self.data)

    def __init_LandUse21(self):

        split_flag, split_path = self.check_split()
        if not split_flag:
            split.split_landuse_dataset(mat_path=self.config.dataset_dir, output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)
        self.data = self.normalize_data(self.data)

    def __init_Prokaryotic(self):

        split_flag, split_path = self.check_split()
        if not split_flag:
            split.split_prokaryotic_dataset(mat_path=self.config.dataset_dir, output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)

    def __init_Scene15(self):

        split_flag, split_path = self.check_split()
        if not split_flag:
            split.split_scene15_dataset(mat_path=self.config.dataset_dir, output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)
        self.data = self.normalize_data(self.data)

    def __init_Reuters(self):
        split_flag, split_path = self.check_split()
        if not split_flag:
            split.split_Reuters_dataset(mat_path=self.config.dataset_dir, output_path=split_path)

        mat_content = sio.loadmat(str(split_path))
        self.data, self.labels = self.get_data_and_labels(self.num_views, mat_content)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        sample = []
        for i in range(self.num_views):
            # 获取每个视图的数据
            view_data = self.data[i][index]
            sample.append(view_data)
        label = self.labels[index]

        return sample, label

    def get_num_classes(self):
        """返回数据集的类别数量"""
        return len(torch.unique(self.labels))

    def get_feature_dimensions(self):
        """返回每个视图的特征维度"""
        feature_dims = []

        for i in range(self.num_views):
            feature_dims.append(self.data[i].shape[1])

        return feature_dims

def get_loader(config,mode,shuffle=True):
    dataset=MMDataset(config=config,mode=mode)
    feature_dims=dataset.get_feature_dimensions()
    config.feature_dims=feature_dims
    config.data_len=len(dataset)
    if config.dataset == 'Synthetic':
        if mode == 'train':
            config.n_train = len(dataset)
        elif mode == 'test':
            config.n_test = len(dataset)
    else:
        if mode == 'train':
            config.n_train = len(dataset)
        elif mode == 'valid':
            config.n_test = len(dataset)
        elif mode == 'test':
            config.n_test = len(dataset)


    return DataLoader(dataset=dataset,
                      batch_size=config.batch_size,
                      shuffle=shuffle,
                      drop_last=False,
                      generator=torch.Generator(device=config.device_name))











