import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        """
        单视图有用编码器
        Args:
            input_dim: 输入维度 D(v)
            latent_dim: 潜在维度 l
            hidden_dims: 隐藏层维度列表
        """
        super(SingleViewEncoder, self).__init__()

        self.dropout_rata = 0.3
        # 构建三层全连接网络
        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for i,hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rata))
            prev_dim = hidden_dim

        # 添加输出层（潜在表示层）
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据 [batch_size, input_dim]
        Returns:
            representation: 输出表示 [batch_size, latent_dim]
        """
        return self.encoder(x)


class MultiViewUsefulEncoderSystem(nn.Module):
    def __init__(self, config):
        """
        多视图有用编码器系统
        Args:
            feature_dims: 各视图的输入维度列表 [D(1), D(2), ..., D(V)]
            latent_dim: 共享潜在维度 l
            hidden_dims: 隐藏层维度
        """
        super(MultiViewUsefulEncoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # 为每个视图创建专用的编码器
        self.encoders = nn.ModuleList([
            SingleViewEncoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, x_list):
        """
        前向传播所有视图
        Args:
            x_list: 各视图输入数据列表 [X(1), X(2), ..., X(V)]
        Returns:
            useful_representations: 各视图的有用表示列表
        """
        useful_representations = []

        for i, (x, encoder) in enumerate(zip(x_list, self.encoders)):
            useful_v = encoder(x)  # useful(v) = e(v)_useful(X(v))
            useful_representations.append(useful_v)

        return useful_representations


class MultiViewTrashEncoderSystem(nn.Module):
    def __init__(self, config):
        """
        多视图有用编码器系统
        Args:
            feature_dims: 各视图的输入维度列表 [D(1), D(2), ..., D(V)]
            latent_dim: 共享潜在维度 l
            hidden_dims: 隐藏层维度
        """
        super(MultiViewTrashEncoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # 为每个视图创建专用的编码器
        self.encoders = nn.ModuleList([
            SingleViewEncoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, x_list):
        """
        前向传播所有视图
        Args:
            x_list: 各视图输入数据列表 [X(1), X(2), ..., X(V)]
        Returns:
            trash_representations: 各视图的有用表示列表
        """
        trash_representations = []

        for i, (x, encoder) in enumerate(zip(x_list, self.encoders)):
            trash_v = encoder(x)
            trash_representations.append(trash_v)

        return trash_representations

    def encode_single_view(self, x, view_idx):
        """
        编码单个视图
        Args:
            x: 单个视图输入数据
            view_idx: 视图索引
        Returns:
            trash_representation: 单个视图的有用表示
        """
        return self.encoders[view_idx](x)


class TreasureRepresentationEncoder(nn.Module):
    def __init__(self, config):
        super(TreasureRepresentationEncoder, self).__init__()
        self.latent_dim = int(config.latent_dim)
        self.num_views = int(config.num_views)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=4,
            batch_first=True
        )

        self.transform = nn.Sequential(
            nn.Linear(self.latent_dim , self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Tanh()
        )

    def forward(self, trash_representations):

        stacked_trash = torch.stack(trash_representations, dim=1)
        # 使用注意力机制
        attended_trash, _ = self.attention(
            stacked_trash, stacked_trash, stacked_trash
        )
        # 平均池化
        mean_attended = torch.mean(attended_trash, dim=1)

        treasure = self.transform(mean_attended)

        return treasure

class PredictionFusion(nn.Module):
    def __init__(self, config):
        super( PredictionFusion, self).__init__()
        self.latent_dim = int(config.latent_dim)
        self.num_classes = int(config.num_classes)
        self.num_views = config.num_views
        self.hidden_dims = config.hidden_dims

        prev_dim = self.latent_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=4,
            batch_first=True
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Sigmoid()
        )

        layers = []

        for i,hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, useful_representations, treasure):

        stacked_useful = torch.stack(useful_representations, dim=1)

        treasure_query = treasure.unsqueeze(1)
        attended_useful, _ = self.cross_attention(
            treasure_query, stacked_useful, stacked_useful
        )
        attended_useful = attended_useful.squeeze(1)

        gate = self.fusion_gate(torch.cat([attended_useful, treasure], dim=1))
        fused_representation = gate * attended_useful + (1 - gate) * treasure


        logits = self.classifier(fused_representation)

        return logits

