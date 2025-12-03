import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

from data_pipeline import get_datasets

def get_true_and_pred(model, test_ds):
    """
    Iterate over a tf.data dataset and return (y_true, y_pred_probs)
    Handles datasets yielding either (x, y) or (x, y, sample_weight)
    Returns:
      y_true: np.array of integer class labels
      y_pred_probs: np.array of shape (N, C) with predicted probabilities
    """
    y_trues = []
    y_preds = []

    for batch in test_ds:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            continue

        probs = model.predict_on_batch(x)

        if len(y.shape) > 1 and y.shape[-1] > 1:
            y_int = np.argmax(y.numpy(), axis=1)
        else:
            y_int = y.numpy().reshape(-1)

        y_trues.append(y_int)
        y_preds.append(probs)

    y_true = np.concatenate(y_trues, axis=0)
    y_pred_probs = np.concatenate(y_preds, axis=0)
    return y_true, y_pred_probs


def compute_per_class_specificity(cm):
    """Given a confusion matrix cm (C x C), return specificity per class."""
    specificity = []
    cm = np.array(cm)
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return np.array(specificity)


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_prob, class_names, out_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, output_path, alpha=0.4):
    import cv2

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    cv2.imwrite(output_path, superimposed_img)


def evaluate_and_visualize(model_path, test_csv, batch_size=16, class_names=None, reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "gradcam"), exist_ok=True)

    print("Loading model:", model_path)
    model = load_model(model_path)

    print("Preparing dataset from CSV (test):", test_csv)
    train_ds, val_ds, test_ds = get_datasets(
        train_csv="dataset/processed/train.csv",
        val_csv="dataset/processed/val.csv",
        test_csv=test_csv,
        batch_size=batch_size,
        augment=False
    )

    if class_names is None:
        if hasattr(test_ds, "class_names"):
            class_names = list(test_ds.class_names)
        else:
            n_classes = int(model.output_shape[-1])
            class_names = [str(i) for i in range(n_classes)]

    print("Class names:", class_names)

    print("Computing predictions on test set...")
    y_true, y_pred_probs = get_true_and_pred(model, test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print("Saved confusion matrix to:", cm_path)

    specs = compute_per_class_specificity(cm)
    for i, name in enumerate(class_names):
        print(f"Class {name}: Sensitivity(Recall) = {cm[i,i] / (cm[i,:].sum() + 1e-8):.4f}, Specificity = {specs[i]:.4f}")

    roc_path = os.path.join(reports_dir, "roc_multiclass.png")
    plot_roc(y_true, y_pred_probs, class_names, roc_path)
    print("Saved ROC plot to:", roc_path)

    print("Generating Grad-CAM visualizations (up to 10 samples)...")
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        print("Could not find a Conv2D layer for Grad-CAM. Skipping Grad-CAM." )
        return

    print("Using last conv layer:", last_conv_layer)

    sample_count = 0
    for batch in test_ds:
        if sample_count >= 10:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            continue

        probs = model.predict_on_batch(x)
        preds = np.argmax(probs, axis=1)

        for i in range(x.shape[0]):
            if sample_count >= 10:
                break
            img_tensor = x[i:i+1]
            img_arr = (img_tensor.numpy()[0] * 255.0).astype(np.uint8)

            pred_index = int(preds[i])
            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer, pred_index)
            out_file = os.path.join(reports_dir, "gradcam", f"sample_{sample_count}_pred_{pred_index}.jpg")
            save_and_display_gradcam(img_arr, heatmap, out_file)
            sample_count += 1

    print(f"Saved {sample_count} Grad-CAM images to {os.path.join(reports_dir, 'gradcam')}")

if __name__ == "__main__":
    evaluate_and_visualize('models/final_model.h5', 'dataset/processed/test.csv', batch_size=16, reports_dir='reports')
