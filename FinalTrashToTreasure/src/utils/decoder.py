import torch
import torch.nn as nn


class SingleViewDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        """
        Single-view decoder
        Args:
            output_dim: Output dimension D(v) (original data dimension)
            latent_dim: Latent dimension l (input dimension is 2l)
            hidden_dims: Hidden layer dimensions list (reverse order)
        """
        super(SingleViewDecoder, self).__init__()

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.input_dim = 2 * latent_dim  # T(v) = [useful; trash] ∈ R^(2l)
        self.dropout_rata=0.3

        # Build three-layer fully connected decoder network
        layers = []
        prev_dim = self.input_dim

        # Add hidden layers (reverse order)
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Add Dropout layer (except the last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rata))
            prev_dim = hidden_dim

        # Add output layer (reconstruct original data)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, t_representation):
        """
        Forward pass: Reconstruct original data from refined representation
        Args:
            t_representation: Refined representation T(v) [batch_size, 2*l]
        Returns:
            x_reconstructed: Reconstructed data X̂(v) [batch_size, output_dim]
        """

        return self.decoder(t_representation)


class MultiViewDecoderSystem(nn.Module):
    def __init__(self, config):
        """
        Multi-view decoder system
        Args:
            view_dims: Original dimension list for each view [D(1), D(2), ..., D(V)]
            latent_dim: Latent dimension l
            hidden_dims: Hidden layer dimensions (reverse order)
        """
        super(MultiViewDecoderSystem, self).__init__()

        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.feature_dims = config.feature_dims
        self.num_views = len(self.feature_dims)

        # Create dedicated decoder for each view
        self.decoders = nn.ModuleList([
            SingleViewDecoder(dim, self.latent_dim, self.hidden_dims)
            for dim in self.feature_dims
        ])

    def forward(self, t_representations):
        """
        Forward pass for all views
        Args:
            t_representations: Refined representation list for each view [T(1), T(2), ..., T(V)]
        Returns:
            x_reconstructed: Reconstructed data list for each view [X̂(1), X̂(2), ..., X̂(V)]
        """
        reconstructed_data = []

        for i, (t_rep, decoder) in enumerate(zip(t_representations, self.decoders)):
            x_hat_v = decoder(t_rep)  # X̂(v) = d(v)(T(v))
            reconstructed_data.append(x_hat_v)

        return reconstructed_data

    def decode_single_view(self, t_representation, view_idx):
        """
        Decode single view
        Args:
            t_representation: Refined representation of single view
            view_idx: View index
        Returns:
            x_reconstructed: Reconstructed data of single view
        """
        return self.decoders[view_idx](t_representation)