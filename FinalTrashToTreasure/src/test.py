import os
import sys
import torch
import argparse

from config import get_config
from data_loader import get_loader
from model import TrashToTreasure
from utils.eval_metrice import eval_senti
from main import set_seed


def load_model(config, model_path='pre_trained_models/best_model.pt'):
    """
    Load pre-trained model
    
    Args:
        config: Configuration object
        model_path: Model file path (can be relative or absolute path)
    
    Returns:
        Model with loaded weights
    """
    # Handle relative paths: if path doesn't exist, try to find in src directory
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        # Try to find in src directory
        src_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(src_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}\nPlease ensure the model file is saved in the correct location.")
    
    # Check necessary configuration attributes
    if not hasattr(config, 'feature_dims') or config.feature_dims is None:
        raise AttributeError(
            "config.feature_dims is not set. Please call get_loader() to load data and set this attribute first.\n"
            "Model creation requires knowing the feature dimensions of each view."
        )
    
    # Create model instance
    model = TrashToTreasure(config).to(config.device)
    
    # Load model weights
    print(f"Loading model: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=config.device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}\nPlease check if the model file is complete and matches the current model architecture.")
    
    return model


def test_model(config, model_path='pre_trained_models/best_model.pt', 
               test_mode='test', verbose=True):
    """
    Test pre-trained model
    
    Args:
        config: Configuration object
        model_path: Model file path
        test_mode: Test mode ('test', 'valid', 'train')
        verbose: Whether to print detailed information
    
    Returns:
        Test results dictionary containing accuracy, F1 score and other metrics
    """
    # Load data first to set config.feature_dims (required for model creation)
    if verbose:
        print(f"Loading {test_mode} data...")
    test_loader = get_loader(config, mode=test_mode, shuffle=False)
    
    # Now we can safely create and load the model
    model = load_model(config, model_path)
    
    if verbose:
        print(f"{test_mode} data loading completed! Number of samples: {len(test_loader.dataset)}")
    
    # Perform prediction
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data_list, labels) in enumerate(test_loader):
            # Move data to device
            data_list = [x.to(config.device) for x in data_list]
            labels = labels.to(config.device)
            
            # Forward pass
            outputs = model(data_list)
            predictions = outputs['prediction']
            
            # Collect results
            all_predictions.append(predictions)
            all_labels.append(labels)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate results from all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Evaluate results
    if verbose:
        print(f"\n{'='*50}")
        print(f"{test_mode.upper()} Set Evaluation Results:")
        print(f"{'='*50}")
    
    accuracy = eval_senti(all_predictions, all_labels, exclude_zero=False)
    
    if verbose:
        print(f"{'='*50}\n")
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def test_with_custom_args(dataset='Scene15', model_path='pre_trained_models/best_model.pt',
                          test_mode='test', device_name=None, seed=37):
    """
    Test with custom arguments
    
    Args:
        dataset: Dataset name
        model_path: Model file path
        test_mode: Test mode ('test', 'valid', 'train')
        device_name: Device name ('cuda' or 'cpu'), if None then auto-select
        seed: Random seed
    
    Returns:
        Test results dictionary
    """
    # Use sys.argv to simulate command line arguments
    original_argv = sys.argv.copy()
    
    try:
        # Build command line arguments
        test_args = [
            '--dataset', dataset,
            '--seed', str(seed),
        ]
        if device_name:
            test_args.extend(['--device_name', device_name])
        
        # Temporarily replace sys.argv
        sys.argv = ['test.py'] + test_args
        
        # Get configuration (will automatically parse command line arguments)
        config = get_config()
        
        # Set learning_rate (if it doesn't exist)
        if not hasattr(config, 'learning_rate'):
            config.learning_rate = config.c
        
        # Set random seed
        set_seed(config.seed, config.device_name)
        
        # Perform testing
        results = test_model(config, model_path, test_mode, verbose=True)
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    return results


def main():
    """
    Main test function
    Can specify test configuration via command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test pre-trained model')
    parser.add_argument('--dataset', type=str, default='Scene15',
                       choices=['Synthetic', '3Sources', 'Reuters', 'LandUse21', 
                               'Prokaryotic', 'Handwritten', 'Scene15'],
                       help='Dataset name (default: Scene15)')
    parser.add_argument('--model_path', type=str, default='pre_trained_models/best_model.pt',
                       help='Model file path (default: pre_trained_models/best_model.pt)')
    parser.add_argument('--test_mode', type=str, default='test',
                       choices=['test', 'valid', 'train'],
                       help='Test mode (default: test)')
    parser.add_argument('--device_name', type=str, default=None,
                       help='Device name (cuda/cpu), if None then auto-select')
    parser.add_argument('--seed', type=int, default=37,
                       help='Random seed (default: 37)')
    
    args = parser.parse_args()
    
    # Execute test
    print("="*60)
    print("Starting pre-trained model testing")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"Test mode: {args.test_mode}")
    print(f"Device: {args.device_name if args.device_name else ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    results = test_with_custom_args(
        dataset=args.dataset,
        model_path=args.model_path,
        test_mode=args.test_mode,
        device_name=args.device_name,
        seed=args.seed
    )
    
    print("\nTesting completed!")
    print(f"Final accuracy: {results['accuracy']:.4f}")
    
    return results


if __name__ == '__main__':
    main()

