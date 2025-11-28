# ...existing code...
import os
# reduce TF verbose logs before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from src.preprocessing import create_data_generators
from src.model_builder import build_custom_cnn, build_transfer_learning_model


# ---------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------
def train_model(model, train_generator, valid_generator, model_name='model', epochs=50, patience=10):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    os.makedirs('models', exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    print("\n🚀 Training Started...\n")
    start = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks
    )

    duration = time.time() - start
    print(f"\n✅ Training completed in {duration:.2f} seconds\n")

    return history, duration


# ---------------------------------------------------------
# PLOT TRAINING GRAPHS
# ---------------------------------------------------------
def plot_training_history(history, model_name='Model'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()

    # Precision
    axes[1, 0].plot(history.history.get('precision', []), label='Train')
    axes[1, 0].plot(history.history.get('val_precision', []), label='Val')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()

    # Recall
    axes[1, 1].plot(history.history.get('recall', []), label='Train')
    axes[1, 1].plot(history.history.get('val_recall', []), label='Val')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig(f"models/{model_name}_training_history.png")
    plt.show()


# ---------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='custom',
                        choices=['custom', 'mobilenet', 'resnet', 'efficient'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    # Dataset paths
    train_dir = 'data/classification_dataset/train'
    valid_dir = 'data/classification_dataset/valid'
    test_dir = 'data/classification_dataset/test'

    # Load generators
    train_gen, valid_gen, test_gen = create_data_generators(
        train_dir, valid_dir, test_dir, batch_size=args.batch_size
    )

    # ---------------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------------
    if args.model == 'custom':
        model = build_custom_cnn()
        model_name = "custom_cnn"

    elif args.model == "mobilenet":
        model = build_transfer_learning_model(base="mobilenet")
        model_name = "mobilenet"

    elif args.model == "resnet":
        model = build_transfer_learning_model(base="resnet")
        model_name = "resnet"

    elif args.model == "efficient":
        model = build_transfer_learning_model(base="efficient")
        model_name = "efficientnet"

    # ---------------------------------------------------------
    # TRAIN THE MODEL
    # ---------------------------------------------------------
    history, duration = train_model(
        model,
        train_gen,
        valid_gen,
        model_name=model_name,
        epochs=args.epochs
    )

    # ---------------------------------------------------------
    # PLOT RESULTS
    # ---------------------------------------------------------
    plot_training_history(history, model_name=model_name)