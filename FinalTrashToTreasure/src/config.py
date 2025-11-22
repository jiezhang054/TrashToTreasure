import argparse
import pprint
from pathlib import Path

import torch
from torch import optim

# Global directories
username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {
    'Synthetic':data_dir.joinpath('new_synthetic.mat'),
    '3Sources': data_dir.joinpath('3-sources.mat'),
    'Handwritten':data_dir.joinpath('handwritten.mat'),
    'LandUse21':data_dir.joinpath('LandUse-21.mat'),
    'Prokaryotic':data_dir.joinpath('new_prokaryotic.mat'),
    'Scene15':data_dir.joinpath('Scene-15.mat'),
    'Reuters':data_dir.joinpath('Reuters.mat'),
}

num_views ={
    'Synthetic':2,
    '3Sources': 3,
    'Handwritten':6,
    'LandUse21':3,
    'Prokaryotic':3,
    'Scene15':3,
    'Reuters':5,
}

num_classes={
    'Synthetic':10,
    '3Sources': 6,
    'Handwritten':10,
    'LandUse21':21,
    'Prokaryotic':4,
    'Scene15':15,
    'Reuters':6,
}

# Dictionaries for optimizers and activation functions
optimizer_dict = {
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}

output_dim_dict = {
    'Synthetic':1,
    '3Sources': 1,
    'Handwritten':1,
    'LandUse21':1,
    'Prokaryotic':1,
    'Scene15':1,
    'Reuters':1,
}

criterion_dict = {
    'Synthetic':'CrossEntropyLoss',
    '3Sources': 'CrossEntropyLoss',
    'Handwritten':'CrossEntropyLoss',
    'LandUse21':'CrossEntropyLoss',
    'Prokaryotic':'CrossEntropyLoss',
    'Scene15':'CrossEntropyLoss',
    'Reuters':'CrossEntropyLoss'
}


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Trash to Treasure')

    # Dataset and paths
    parser.add_argument('--dataset', type=str, default='Scene15',
                        choices=['Synthetic', '3Sources','Reuters','LandUse21','Prokaryotic','Handwritten','Scene15'],
                        help='Dataset to use (default: Synthetic)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='Path for storing the dataset')

    # Model architecture parameters
    parser.add_argument('--multiseed', action='store_true',
                        help='Training using multiple seeds')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256],)


    # Training settings
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Initial learning rate for  model parameters (default: 5e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='L2 penalty factor of the  Adam optimizer')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='Optimizer to use (default: Adam)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clip value (default: 1.0)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--when', type=int, default=20,
                        help='When to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=30,
                        help='When to stop training if best does not change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='Update batch interval')
    parser.add_argument('--sigma', type=float, default=1,choices=[0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100],
                        help='useful_loss weight')
    parser.add_argument('--beta', type=float, default=1,choices=[0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100],
                        help='gap_loss weight')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Frequency of result logging (default: 100)')


    parser.add_argument('--seed', type=int, default=37,
                        help='Random seed (default: 37)')
    parser.add_argument('--device_name', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use during training (default: cuda if available)')

    args = parser.parse_args()
    return args


class Config:
    """Configuration class to store and manage parameters."""

    def __init__(self, args):
        """Initialize the configuration with command-line arguments."""
        self.__dict__.update(vars(args))
        # Update class attributes with argparse arguments
        self.dataset = str(args.dataset.strip())  # Normalize dataset name
        self.dataset_dir = data_dict[self.dataset]  # Dataset directory
        self.num_classes = num_classes[self.dataset]
        self.num_views = num_views[self.dataset]
        self.device = torch.device(args.device_name)  # Device (CPU or GPU)


    def to_dict(self):
        """Convert Config object to dictionary."""
        return self.__dict__

    def __str__(self):
        """Pretty-print configurations in alphabetical order."""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config():
    """Get and return the configuration object."""
    args = get_args()
    config = Config(args)
    return config


if __name__ == "__main__":
    config = get_config()
    print(config)
