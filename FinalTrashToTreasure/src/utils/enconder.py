import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        """
        Single-view useful encoder
        Args:
            input_dim: Input dimension D(v)
            latent_dim: Latent dimension l
            hidden_dims: Hidden layer dimensions list
        """
        super(SingleViewEncoder, self).__init__()

        self.dropout_rata = 0.3
        # Build three-layer fully connected network
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for i,hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rata))
            prev_dim = hidden_dim

        # Add output layer (latent representation layer)
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input data [batch_size, input_dim]
        Returns:
            representation: Output representation [batch_size, latent_dim]
        """
        return self.encoder(x)


class MultiViewUsefulEncoderSystem(nn.Module):
    def __init__(self, config):
        """
        Multi-view useful encoder system
        Args:
            feature_dims: Input dimension list for each view [D(1), D(2), ..., D(V)]
            latent_dim: Shared latent dimension l
            hidden_dims: Hidden layer dimensions
        """
        super(MultiViewUsefulEncoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # Create dedicated encoder for each view
        self.encoders = nn.ModuleList([
            SingleViewEncoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, x_list):
        """
        Forward pass for all views
        Args:
            x_list: Input data list for each view [X(1), X(2), ..., X(V)]
        Returns:
            useful_representations: Useful representation list for each view
        """
        useful_representations = []

        for i, (x, encoder) in enumerate(zip(x_list, self.encoders)):
            useful_v = encoder(x)  # useful(v) = e(v)_useful(X(v))
            useful_representations.append(useful_v)

        return useful_representations


class MultiViewTrashEncoderSystem(nn.Module):
    def __init__(self, config):
        """
        Multi-view trash encoder system
        Args:
            feature_dims: Input dimension list for each view [D(1), D(2), ..., D(V)]
            latent_dim: Shared latent dimension l
            hidden_dims: Hidden layer dimensions
        """
        super(MultiViewTrashEncoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # Create dedicated encoder for each view
        self.encoders = nn.ModuleList([
            SingleViewEncoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, x_list):
        """
        Forward pass for all views
        Args:
            x_list: Input data list for each view [X(1), X(2), ..., X(V)]
        Returns:
            trash_representations: Trash representation list for each view
        """
        trash_representations = []

        for i, (x, encoder) in enumerate(zip(x_list, self.encoders)):
            trash_v = encoder(x)
            trash_representations.append(trash_v)

        return trash_representations

    def encode_single_view(self, x, view_idx):
        """
        Encode single view
        Args:
            x: Single view input data
            view_idx: View index
        Returns:
            trash_representation: Trash representation of single view
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
        # Use attention mechanism
        attended_trash, _ = self.attention(
            stacked_trash, stacked_trash, stacked_trash
        )
        # Average pooling
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

