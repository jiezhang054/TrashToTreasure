import torch
import torch.nn as nn


class SingleViewDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        """
        单视图解码器
        Args:
            output_dim: 输出维度 D(v)（原始数据维度）
            latent_dim: 潜在维度 l（输入维度是 2l）
            hidden_dims: 隐藏层维度列表（逆序）
        """
        super(SingleViewDecoder, self).__init__()

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.input_dim = 2 * latent_dim  # T(v) = [useful; trash] ∈ R^(2l)
        self.dropout_rata=0.3

        # 构建三层全连接解码网络
        layers = []
        prev_dim = self.input_dim

        # 添加隐藏层（逆序）
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            # 添加Dropout层（除了最后一层）
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rata))
            prev_dim = hidden_dim

        # 添加输出层（重构原始数据）
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, t_representation):
        """
        前向传播：从精炼表示重构原始数据
        Args:
            t_representation: 精炼表示 T(v) [batch_size, 2*l]
        Returns:
            x_reconstructed: 重构数据 X̂(v) [batch_size, output_dim]
        """

        return self.decoder(t_representation)


class MultiViewDecoderSystem(nn.Module):
    def __init__(self, config):
        """
        多视图解码器系统
        Args:
            view_dims: 各视图的原始维度列表 [D(1), D(2), ..., D(V)]
            latent_dim: 潜在维度 l
            hidden_dims: 隐藏层维度（逆序）
        """
        super(MultiViewDecoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # 为每个视图创建专用的解码器
        self.decoders = nn.ModuleList([
            SingleViewDecoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, t_representations):
        """
        前向传播所有视图
        Args:
            t_representations: 各视图的精炼表示列表 [T(1), T(2), ..., T(V)]
        Returns:
            x_reconstructed: 各视图的重构数据列表 [X̂(1), X̂(2), ..., X̂(V)]
        """
        reconstructed_data = []

        for i, (t_rep, decoder) in enumerate(zip(t_representations, self.decoders)):
            x_hat_v = decoder(t_rep)  # X̂(v) = d(v)(T(v))
            reconstructed_data.append(x_hat_v)

        return reconstructed_data

    def decode_single_view(self, t_representation, view_idx):
        """
        解码单个视图
        Args:
            t_representation: 单个视图的精炼表示
            view_idx: 视图索引
        Returns:
            x_reconstructed: 单个视图的重构数据
        """
        return self.decoders[view_idx](t_representation)