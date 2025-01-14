"""
Author: anish.ravishankar@kaplan.com
This script trains a CNN model for image classification using TensorFlow and Keras.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from utils import load_images_from_folder


class CNNModel:
    """
    A CNN model class that handles model definition and training.
    """
    
    def __init__(self, input_shape=None):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Optional tuple defining the input shape for immediate model building
        """
        self.model = None
        if input_shape:
            self.build_model(input_shape)
    
    def build_model(self, input_shape):
        """
        Build the CNN model architecture.
        
        Args:
            input_shape: Tuple defining the input shape (height, width, channels)
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='linear')  # Regression output for the count
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             batch_size=32, epochs=10):
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built before training. Call build_model() first.")
            
            
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
        return history
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
            
        self.model.save(filepath)
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be built before making predictions.")
            
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model must be built before evaluation.")
            
        loss, mae = self.model.evaluate(X_test, y_test)
        return {
            'loss': loss,
            'mae': mae
        }

class DataAugmentation:
    def __init__(self, rotation_range=0, width_shift_range=0, height_shift_range=0, 
                 zoom_range=0, darkening_factor=1):
        self.data_gen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range
        )
        self.darkening_factor = darkening_factor

    def generate_augmented_images(self, X, y, num_augmented):
        augmented_images = []
        augmented_labels = []
        

        for i in range(len(X)):
            img = X[i]
            label = y[i]
            img = np.expand_dims(img, axis=0)
            
            # Generate augmented versions
            aug_iter = self.data_gen.flow(img, np.array([label]), batch_size=1)
            for _ in range(num_augmented):
                aug_img = next(aug_iter)[0]
                # Ensure aug_img has the same shape as original images
                aug_img = aug_img.reshape(X[0].shape)
                augmented_images.append(aug_img)
                augmented_labels.append(label)

        # Convert augmented data to numpy arrays
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        return augmented_images, augmented_labels

    def darken_images(self, X, y):
        darkened_images = []
        darkened_labels = []

        for i in range(len(X)):
            darkened_img = X[i] * self.darkening_factor  # Darken pixels
            darkened_images.append(darkened_img)
            darkened_labels.append(y[i])

        # Convert darkened data to numpy arrays
        darkened_images = np.array(darkened_images)
        darkened_labels = np.array(darkened_labels)

        return darkened_images, darkened_labels

    def generate_augmented_and_darkened_images(self, X, y, num_augmented):
        """Combines both augmentation and darkening in one step"""
        # First generate augmented images
        aug_images, aug_labels = self.generate_augmented_images(X, y, num_augmented)
        
        # Then darken the augmented images
        darkened_aug_images, darkened_aug_labels = self.darken_images(aug_images, aug_labels)
        
        return darkened_aug_images, darkened_aug_labels


if __name__ == '__main__':
    folder_path = "SyntheticImages"  

    # Load and preprocess the images
    X, y = load_images_from_folder(folder_path)

    # Data augmentation
    data_augmentation = DataAugmentation(rotation_range=1, width_shift_range=0.05, height_shift_range=0.1, zoom_range=0.01, darkening_factor=0.2)
    augmented_images, augmented_labels = data_augmentation.generate_augmented_and_darkened_images(X, y, num_augmented=50)

    # Combine original, augmented, and darkened data
    X_combined = np.concatenate([X, augmented_images], axis=0)
    y_combined = np.concatenate([y, augmented_labels], axis=0)
    
    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build and train the CNN model
    model = CNNModel(input_shape=(64, 128, 1))

    # Train the model
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=32,
        epochs=10 # Current model is trained for 100 epochs; you can adjust this value
    )

    # Evaluate and make predictions
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Test loss: {eval_metrics['loss']}, Test MAE: {eval_metrics['mae']}")
    
    # Save the model
    model.save_model('model_test.h5')
