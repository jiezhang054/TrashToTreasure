# TrashToTreasure: An Informative and Interactive Multi-View Classification Framework

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **TrashToTreasure: An Informative and Interactive Multi-View Classification Framework**. The framework introduces a novel approach to multi-view learning by separating useful and trash information from different views, then transforming trash into treasure through an interactive mechanism.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

TrashToTreasure is a multi-view classification framework that:

- **Separates Information**: Distinguishes between useful and trash information from multiple views
- **Transforms Trash to Treasure**: Converts less informative trash representations into valuable treasure representations through an interactive attention mechanism
- **Reconstructs Views**: Uses a decoder system to reconstruct original views from refined representations
- **Fuses Predictions**: Combines useful representations and treasure for final classification

## ✨ Features

- **Multi-View Encoder System**: Separate encoders for useful and trash information extraction
- **Treasure Representation Learning**: Interactive attention mechanism to transform trash into treasure
- **Reconstruction Loss**: Ensures information preservation through view reconstruction
- **Gap Loss**: Maximizes the gap between treasure and trash representations
- **Useful Loss**: Supervises useful representation learning
- **Flexible Architecture**: Supports multi-view datasets with different numbers of views

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 12.1+ (for GPU support, optional)
- Conda

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd FinalTrashToTreasure
```

2. **Create and activate conda environment using the provided environment file:**
```bash
conda env create -f treasure_environment.yml
conda activate treasure
```

The `treasure_environment.yml` file contains all necessary dependencies including:
- PyTorch 2.4.1 with CUDA 12.1 support
- scikit-learn, scipy, numpy for data processing
- matplotlib for visualization
- All other required packages

**Note**: If you don't have CUDA support, you can modify the environment file or install PyTorch CPU version separately.

## 🏃 Quick Start

### Training

Train the model on Scene15 dataset:

```bash
cd src
python main.py --dataset Scene15 --batch_size 64 --num_epochs 100
```

### Testing

Test a pre-trained model:

```bash
cd src
python test.py --dataset Scene15 --model_path pre_trained_models/best_model.pt --test_mode test
```

## 💻 Usage

### Training Parameters

Key training parameters for Scene15 dataset:

```bash
python main.py \
    --dataset Scene15 \              # Dataset name
    --batch_size 64 \                # Batch size
    --latent_dim 32 \                # Latent dimension
    --hidden_dims 128 256 \          # Hidden layer dimensions
    --c 5e-3 \                       # Learning rate
    --weight_decay 5e-5 \            # Weight decay
    --sigma 1.0 \                    # Reconstruction loss weight
    --beta 1.0 \                     # Gap loss weight
    --num_epochs 100 \               # Number of epochs
    --patience 30 \                  # Early stopping patience
    --seed 37 \                      # Random seed
    --device_name cuda               # Device (cuda/cpu)
```

### Testing Parameters

```bash
python test.py \
    --dataset Scene15 \              # Dataset name
    --model_path pre_trained_models/best_model.pt \  # Model path
    --test_mode test \               # Test mode (test/valid/train)
    --device_name cuda               # Device (cuda/cpu)
```

### Programmatic Usage

```python
from src.test import test_with_custom_args

# Test with custom arguments
results = test_with_custom_args(
    dataset='Scene15',
    model_path='pre_trained_models/best_model.pt',
    test_mode='test',
    device_name='cuda',
    seed=37
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

## 📁 Project Structure

```
FinalTrashToTreasure/
├── src/
│   ├── main.py                 # Main training script
│   ├── test.py                 # Testing script
│   ├── model.py                # TrashToTreasure model definition
│   ├── solver.py               # Training and evaluation solver
│   ├── config.py               # Configuration management
│   ├── data_loader.py          # Data loading utilities
│   ├── split.py                # Dataset splitting utilities
│   ├── utils/
│   │   ├── enconder.py         # Encoder modules (useful/trash/treasure)
│   │   ├── decoder.py          # Decoder modules
│   │   ├── loss.py             # Loss functions
│   │   ├── eval_metrice.py     # Evaluation metrics
│   │   └── tools.py             # Utility functions
│   └── pre_trained_models/     # Directory for saved models
├── datasets/                    # Dataset directory
│   ├── Scene-15.mat            # Original Scene15 dataset
│   └── Scene-15_622.mat         # Pre-split Scene15 dataset
├── treasure_environment.yml     # Conda environment file
└── README.md                    # This file
```

## ⚙️ Configuration

### Model Architecture

- **Latent Dimension**: Controls the dimensionality of learned representations (default: 32)
- **Hidden Dimensions**: List of hidden layer sizes in encoders/decoders (default: [128, 256])
- **Number of Views**: 3 views for Scene15 dataset

### Loss Functions

The framework uses four main loss functions:

1. **Classification Loss**: Standard cross-entropy loss for final predictions
2. **Reconstruction Loss**: MSE loss between original and reconstructed views (weighted by `sigma`)
3. **Useful Loss**: Supervised loss for useful representations
4. **Gap Loss**: Maximizes gap between treasure and trash (weighted by `beta`)

Total loss: `L = L_cls + L_useful + σ * L_rec + β * L_gap`

### Training Strategy

- **Optimizer**: Adam optimizer
- **Learning Rate Scheduling**: ReduceLROnPlateau (reduces LR by 0.5 when validation loss plateaus)
- **Early Stopping**: Stops training if test accuracy doesn't improve for `patience` epochs
- **Gradient Clipping**: Clips gradients to prevent explosion (default: 1.0)
- **Gradient Accumulation**: Supports gradient accumulation via `update_batch` parameter

### Dataset

The repository includes the **Scene15** dataset, which contains:
- **3 views**: Different feature representations of scene images
- **15 classes**: Various scene categories
- **Pre-split version**: `Scene-15_622.mat` with train/valid/test splits (60%/20%/20%)

If the pre-split version doesn't exist, the framework will automatically split the original dataset.

## 📈 Results

The model automatically saves the best model based on test accuracy to `src/pre_trained_models/best_model.pt`. Evaluation metrics include:

- **Accuracy**: Classification accuracy
- **Weighted F1 Score**: F1 score weighted by class support
- **Macro F1 Score**: Unweighted mean F1 score across classes
- **AUC**: Area under the ROC curve (multi-class)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and the open-source community
- Dataset providers for making multi-view datasets publicly available

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or increase `update_batch` for gradient accumulation
2. **Environment creation fails**: Ensure you have conda installed and try updating conda first
3. **Model loading error**: Check that the model architecture matches the saved checkpoint
4. **Dataset not found**: Ensure `Scene-15.mat` or `Scene-15_622.mat` is in the `datasets/` directory

### Getting Help

- Check existing Issues for solutions
- Open a new issue with detailed error messages and system information

---

**Note**: This is the official implementation of the TrashToTreasure framework. For detailed methodology and experimental results, please refer to the original paper.
