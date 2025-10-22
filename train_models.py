"""
Training script for signal classification models
Trains CNN, LSTM, and Hybrid CNN-LSTM models
"""

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from models import create_model, ModelTrainer
import json

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def load_dataset(dataset_path):
    """Load the dataset from pickle file"""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X = dataset['X']
    y = dataset['y']
    snr = dataset['snr']
    
    print(f"Dataset loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y, snr, dataset


def prepare_data(X, y, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets"""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples ({100*(1-test_size-val_size):.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({100*val_size:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({100*test_size:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_single_model(model_type, X_train, y_train, X_val, y_val, 
                       input_shape, num_classes, output_dir):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_type} Model")
    print(f"{'='*80}")
    
    # Create model
    model = create_model(model_type, input_shape, num_classes)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Create trainer
    trainer = ModelTrainer(model, model_name=model_type)
    
    # Set checkpoint path
    checkpoint_path = os.path.join(output_dir, f'{model_type}_best_model.h5')
    
    # Train model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=128,
        checkpoint_path=checkpoint_path
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'{model_type}_final_model.h5')
    trainer.save_model(final_model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f'{model_type}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")
    
    return trainer, history


def main():
    """Main training function"""
    # Configuration
    dataset_path = '/home/ubuntu/signal_classification_project/signal_dataset.pkl'
    output_dir = '/home/ubuntu/signal_classification_project/trained_models'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y, snr, dataset = load_dataset(dataset_path)
    
    # Get dataset info
    num_classes = len(dataset['modulations'])
    input_shape = X.shape[1:]  # (2, 128)
    
    print(f"\nDataset Information:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Modulation types: {dataset['modulations']}")
    print(f"  Input shape: {input_shape}")
    print(f"  SNR range: [{snr.min():.1f}, {snr.max():.1f}] dB")
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)
    
    # Save test data for later evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'modulations': dataset['modulations'],
        'idx_to_mod': dataset['idx_to_mod']
    }
    test_data_path = os.path.join(output_dir, 'test_data.pkl')
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"\nTest data saved to {test_data_path}")
    
    # Train models
    model_types = ['CNN', 'LSTM', 'CNN_LSTM']
    results = {}
    
    for model_type in model_types:
        try:
            trainer, history = train_single_model(
                model_type, X_train, y_train, X_val, y_val,
                input_shape, num_classes, output_dir
            )
            
            # Evaluate on test set
            print(f"\n{'='*80}")
            print(f"Evaluating {model_type} on Test Set")
            print(f"{'='*80}")
            test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
            
            # Store results
            results[model_type] = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_val_loss': float(min(history.history['val_loss'])),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'epochs_trained': len(history.history['loss'])
            }
            
            print(f"\n{model_type} Results:")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Best Val Accuracy: {results[model_type]['best_val_accuracy']:.4f}")
            
        except Exception as e:
            print(f"\nError training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall results
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n\nTraining results saved to {results_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    for model_type, result in results.items():
        print(f"\n{model_type}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Best Val Accuracy: {result['best_val_accuracy']:.4f}")
        print(f"  Epochs Trained: {result['epochs_trained']}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n{'='*80}")
        print(f"Best Model: {best_model[0]} with Test Accuracy: {best_model[1]['test_accuracy']:.4f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

