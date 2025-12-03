import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB2
import os

# Ensure TF/Keras uses channels_last (H, W, C)
tf.keras.backend.set_image_data_format("channels_last")

def build_model(img_size=260, num_classes=3, dropout=0.3, base_trainable=False, weights="imagenet"):
    """
    Build EfficientNetB2 classifier.
    - img_size: int (use 260 for B2)
    - num_classes: int
    - dropout: float
    - base_trainable: bool to make backbone trainable initially
    - weights: "imagenet" or None
    """
    img_size = int(img_size)
    input_shape = (img_size, img_size, 3)

    # Debug prints (safe)
    print("DEBUG (model_EfficientNetB2): building with input_shape =", input_shape)
    print("DEBUG: tf.keras.backend.image_data_format() =", tf.keras.backend.image_data_format())

    # Create explicit input tensor to avoid ambiguous channel inference
    inputs = layers.Input(shape=input_shape)

    # Build backbone with the input tensor to force 3 channels
    try:
        backbone = EfficientNetB2(include_top=False, weights=None, input_tensor=inputs)
    except ValueError as e:
        # Helpful fallback if weights mismatch occurs: rethrow with guidance.
        raise ValueError(
            "Failed to load EfficientNetB2 with weights='%s'.\n"
            "This usually means the model input channels do not match the pretrained weights (ImageNet expects 3-channel RGB).\n"
            "Current input_shape=%s. To bypass pretrained weights, set weights=None in build_model().\nOriginal error: %s"
            % (weights, input_shape, e)
        ) from e

    backbone.trainable = base_trainable

    # Build head
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EffNetB2_clutter")

    return model

def unfreeze_backbone(model, n_last_layers=40):
    """
    Robustly unfreeze the last `n_last_layers` trainable layers in the model.
    Works whether the backbone is a nested Model or just a sequence of layers.

    Strategy:
    - collect all layers that have the attribute `trainable`
    - keep the final classification head (Dense) trainable only if it's within the last n
    - set the last n layers trainable, freeze the rest
    """
    # collect all layers that can be toggled
    all_layers = [l for l in model.layers if hasattr(l, "trainable")]
    total = len(all_layers)
    if total == 0:
        print("unfreeze_backbone: No toggleable layers found.")
        return

    start_idx = max(0, total - int(n_last_layers))
    # Freeze layers before start_idx, unfreeze layers from start_idx onward
    for i, layer in enumerate(all_layers):
        layer.trainable = True if i >= start_idx else False

    print(f"unfreeze_backbone: total_layers={total}, unfreezing_from_index={start_idx} (last {n_last_layers} layers).")

def compile_model(model, initial_lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
