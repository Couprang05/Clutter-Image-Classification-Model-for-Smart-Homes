import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 260           # EfficientNetB2 recommended size
BATCH_SIZE = 16          # lower if OOM
EPOCHS_FROZEN = 12
EPOCHS_FINETUNE = 10
DROPOUT = 0.3
NUM_CLASSES = 3
FINE_TUNE_LAST_N = 40    # unfreeze last N backbone layers
INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# import dataset pipeline
from data_pipeline import get_datasets, compute_class_weights

# Always use our EfficientNetB2 model file
from model_EfficientNetB2 import build_model, unfreeze_backbone, compile_model
print("Imported from model_EfficientNetB2.py")


def main():
    print("Preparing datasets...")
    train_ds, val_ds, test_ds = get_datasets(
        train_csv="dataset/processed/train.csv",
        val_csv="dataset/processed/val.csv",
        test_csv="dataset/processed/test.csv",
        batch_size=BATCH_SIZE,
        augment=True
    )

    print("Computing class weights...")
    class_weights = compute_class_weights("dataset/processed/train.csv")
    print("Class weights:", class_weights)

    print("Building model...")
    model = build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, dropout=DROPOUT, base_trainable=False, weights="imagenet")
    model = compile_model(model, initial_lr=INITIAL_LR)
    model.summary()

    # Callbacks for frozen training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "best_frozen.h5"),
                                           monitor="val_loss", save_best_only=True)
    ]

    print("Starting frozen-head training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Fine-tune
    print(f"Unfreezing last {FINE_TUNE_LAST_N} layers of backbone for fine-tuning...")
    unfreeze_backbone(model, n_last_layers=FINE_TUNE_LAST_N)

    model = compile_model(model, initial_lr=FINETUNE_LR)

    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "best_finetuned.h5"),
                                           monitor="val_loss", save_best_only=True)
    ]

    print("Starting fine-tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks_ft,
        class_weight=class_weights
    )

    final_path = os.path.join(CHECKPOINT_DIR, "final_model.h5")
    model.save(final_path)
    print("Saved final model to:", final_path)


if __name__ == "__main__":
    main()
