import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 260

LABEL_COL = "label"
PATH_COL  = "image_path"

def _read_csv_to_df(csv_path):
    df = pd.read_csv(csv_path)
    df[PATH_COL] = df[PATH_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df

def _process_image(path, label, augment=False):
    """
    path: scalar tf.string tensor
    label: scalar int tensor
    returns: (image_tensor float32 [H,W,3], one_hot_label)
    """
    # read file bytes
    image_bytes = tf.io.read_file(path)

    # decode_image handles JPEG/PNG/etc. expand_animations=False avoids 4-D tensors for GIFs
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)

    # Make channel dimension statically known (height/width will be dynamic)
    image.set_shape([None, None, 3])

    # Convert to float32 [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize to target size
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # Optional augmentation
    if augment:
        # random flip
        image = tf.image.random_flip_left_right(image)

        # small color jitter
        image = tf.image.random_brightness(image, 0.06)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_saturation(image, 0.9, 1.1)

        # random crop/scale: compute crop size robustly
        crop_frac = tf.random.uniform([], 0.90, 1.0)
        new_h = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * crop_frac), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * crop_frac), tf.int32)
        # ensure new_h/new_w >= 1
        new_h = tf.maximum(new_h, 1)
        new_w = tf.maximum(new_w, 1)
        # if crop size equals full size, random_crop will still work
        image = tf.image.random_crop(image, size=[new_h, new_w, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # convert label and one-hot encode
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=3)

    return image, label

def _prepare_dataset(df, batch_size=32, shuffle=False, augment=False, repeat=False):
    paths = df[PATH_COL].values
    labels = df[LABEL_COL].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    # Map once and pass augment flag so we avoid an extra map stage
    ds = ds.map(lambda p, l: _process_image(p, l, augment=augment), num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def get_datasets(train_csv="dataset/processed/train.csv",
                 val_csv="dataset/processed/val.csv",
                 test_csv="dataset/processed/test.csv",
                 batch_size=32,
                 augment=True):
    """
    Returns: train_ds, val_ds, test_ds (tf.data.Dataset)
    - train_ds is shuffled and augmented (if augment=True)
    - val_ds and test_ds are deterministic (no augmentation)
    """
    train_df = _read_csv_to_df(train_csv)
    val_df   = _read_csv_to_df(val_csv)
    test_df  = _read_csv_to_df(test_csv)

    train_ds = _prepare_dataset(train_df, batch_size=batch_size, shuffle=True, augment=augment, repeat=False)
    val_ds   = _prepare_dataset(val_df,   batch_size=batch_size, shuffle=False, augment=False, repeat=False)
    test_ds  = _prepare_dataset(test_df,  batch_size=batch_size, shuffle=False, augment=False, repeat=False)

    return train_ds, val_ds, test_ds

def compute_class_weights(train_csv="dataset/processed/train.csv"):
    """
    Compute sklearn-style class weights {class_index: weight}
    """
    df = pd.read_csv(train_csv)
    labels = df[LABEL_COL].values
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    # weight = total / (num_classes * count)
    num_classes = 3
    weights = {int(k): float(total / (num_classes * c)) for k, c in zip(unique, counts)}
    # ensure every class in [0..num_classes-1] has a weight
    for i in range(num_classes):
        if i not in weights:
            weights[i] = 1.0
    return weights
