
# ...existing code...
import os
# silence routine TF logs before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Build a custom CNN for binary classification
    """
    model = tf.models.Sequential([
        # First Convolutional Block
        tf.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2)),
        tf.layers.Dropout(0.25),
        
        # Second Convolutional Block
        tf.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2)),
        tf.layers.Dropout(0.25),
        
        # Third Convolutional Block
        tf.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.layers.BatchNormalization(),
        tf.shapelayers.MaxPooling2D((2, 2)),
        tf.layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        tf.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2)),
        tf.layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        tf.layers.Flatten(),
        tf.layers.Dense(512, activation='relu'),
        tf.layers.BatchNormalization(),
        tf.layers.Dropout(0.5),
        tf.layers.Dense(256, activation='relu'),
        tf.layers.Dropout(0.5),
        tf.layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    return model


def build_transfer_learning_model(base_model_name='MobileNetV2', 
                                  input_shape=(224, 224, 3),
                                  trainable_layers=0):
    """
    Build transfer learning model with pre-trained base
    Options: MobileNetV2, ResNet50, EfficientNetB0
    """
    
    # Load base model using tf.keras.applications
    if base_model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'EfficientNetB0':
        # EfficientNetB0 may not be available on very old TF versions
        try:
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        except AttributeError:
            raise RuntimeError("EfficientNetB0 not available in this TF version. Upgrade TensorFlow or choose another base_model_name.")
    else:
        raise ValueError("Invalid base model name")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Optional: Unfreeze last few layers for fine-tuning
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    # Build complete model
    model = tf.models.Sequential([
        base_model,
        tf.layers.GlobalAveragePooling2D(),
        tf.layers.Dense(256, activation='relu'),
        tf.layers.BatchNormalization(),
        tf.layers.Dropout(0.5),
        tf.layers.Dense(128, activation='relu'),
        tf.layers.Dropout(0.3),
        tf.layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# Test models
if __name__ == "__main__":
    # Custom CNN
    custom_model = build_custom_cnn()
    custom_model.summary()
    print("\n" + "="*50 + "\n")
    
    # Transfer Learning
    transfer_model = build_transfer_learning_model('MobileNetV2')
    transfer_model.summary()
