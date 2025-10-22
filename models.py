"""
Deep Learning Models for Signal Classification
Includes CNN, LSTM, and Hybrid CNN-LSTM architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


class CNNModel:
    """Convolutional Neural Network for signal classification"""
    
    def __init__(self, input_shape, num_classes, model_name='CNN'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self.build_model()
    
    def build_model(self):
        """Build CNN architecture"""
        model = models.Sequential(name=self.model_name)
        
        # Reshape input for Conv1D: (batch, 2, 128) -> (batch, 128, 2)
        model.add(layers.Reshape((self.input_shape[1], self.input_shape[0]), 
                                 input_shape=self.input_shape))
        
        # First convolutional block
        model.add(layers.Conv1D(64, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))
        
        # Second convolutional block
        model.add(layers.Conv1D(128, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))
        
        # Third convolutional block
        model.add(layers.Conv1D(256, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))
        
        # Fourth convolutional block
        model.add(layers.Conv1D(512, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(0.4))
        
        # Fully connected layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the model"""
        return self.model


class LSTMModel:
    """LSTM Network for signal classification"""
    
    def __init__(self, input_shape, num_classes, model_name='LSTM'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self.build_model()
    
    def build_model(self):
        """Build LSTM architecture"""
        model = models.Sequential(name=self.model_name)
        
        # Reshape input for LSTM: (batch, 2, 128) -> (batch, 128, 2)
        model.add(layers.Reshape((self.input_shape[1], self.input_shape[0]), 
                                 input_shape=self.input_shape))
        
        # First LSTM layer
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Second LSTM layer
        model.add(layers.LSTM(256, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Third LSTM layer
        model.add(layers.LSTM(128, return_sequences=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Fully connected layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the model"""
        return self.model


class HybridCNNLSTM:
    """Hybrid CNN-LSTM Network for signal classification"""
    
    def __init__(self, input_shape, num_classes, model_name='CNN_LSTM'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self.build_model()
    
    def build_model(self):
        """Build Hybrid CNN-LSTM architecture"""
        model = models.Sequential(name=self.model_name)
        
        # Reshape input: (batch, 2, 128) -> (batch, 128, 2)
        model.add(layers.Reshape((self.input_shape[1], self.input_shape[0]), 
                                 input_shape=self.input_shape))
        
        # CNN layers for feature extraction
        model.add(layers.Conv1D(64, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Conv1D(128, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Conv1D(256, kernel_size=8, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        # LSTM layers for temporal modeling
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Fully connected layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the model"""
        return self.model


class ModelTrainer:
    """Trainer class for managing model training"""
    
    def __init__(self, model, model_name='model'):
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def get_callbacks(self, checkpoint_path=None):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=128, checkpoint_path=None):
        """Train the model"""
        callbacks = self.get_callbacks(checkpoint_path)
        
        print(f"\nTraining {self.model_name}...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print(f"\nEvaluating {self.model_name}...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def create_model(model_type, input_shape, num_classes):
    """
    Factory function to create models
    
    Args:
        model_type: 'CNN', 'LSTM', or 'CNN_LSTM'
        input_shape: Input shape (channels, samples)
        num_classes: Number of output classes
    
    Returns:
        Model instance
    """
    if model_type == 'CNN':
        model_builder = CNNModel(input_shape, num_classes, model_name='CNN')
    elif model_type == 'LSTM':
        model_builder = LSTMModel(input_shape, num_classes, model_name='LSTM')
    elif model_type == 'CNN_LSTM':
        model_builder = HybridCNNLSTM(input_shape, num_classes, model_name='CNN_LSTM')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_builder.compile_model()
    return model_builder.get_model()


if __name__ == "__main__":
    # Test model creation
    input_shape = (2, 128)  # (I/Q channels, samples)
    num_classes = 11
    
    print("Creating CNN model...")
    cnn_model = create_model('CNN', input_shape, num_classes)
    cnn_model.summary()
    
    print("\n" + "="*80 + "\n")
    
    print("Creating LSTM model...")
    lstm_model = create_model('LSTM', input_shape, num_classes)
    lstm_model.summary()
    
    print("\n" + "="*80 + "\n")
    
    print("Creating Hybrid CNN-LSTM model...")
    hybrid_model = create_model('CNN_LSTM', input_shape, num_classes)
    hybrid_model.summary()

