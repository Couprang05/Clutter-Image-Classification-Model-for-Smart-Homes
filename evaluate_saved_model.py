import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


def detect_label_column(df):
    if 'label' in df.columns:
        return 'label'
    if 'clutter_level' in df.columns:
        return 'clutter_level'
    return df.columns[-1]

def map_string_labels(labels):
    if all(isinstance(x, str) for x in labels):
        uniq = sorted(list(pd.Series(labels).unique()))
        mapping = {lab: i for i, lab in enumerate(uniq)}
        mapped = np.array([mapping[x] for x in labels], dtype=int)
        inv = {v:k for k,v in mapping.items()}
        return mapped, mapping, inv
    else:
        arr = np.array([int(x) for x in labels], dtype=int)
        uniq = sorted(list(set(arr)))
        inv = {i:str(i) for i in uniq}
        mapping = {str(i): i for i in uniq}
        return arr, mapping, inv

def build_dataset(paths, labels, batch_size=32, img_size=260):
    paths = [str(p) for p in paths]
    labels = [int(l) for l in labels]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _read_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(_read_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    try:
        ds = ds.apply(tf.data.experimental.ignore_errors())
    except Exception:
        pass
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def plot_confusion(cm, class_names, out_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_prf(per_class_metrics, class_names, out_path):
    # per_class_metrics: (precision, recall, f1) arrays
    precision, recall, f1 = per_class_metrics
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(8,5))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylabel('Score'); plt.ylim(0,1.05)
    plt.title('Per-class Precision / Recall / F1')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_loss_accuracy(loss, acc, out_path):
    plt.figure(figsize=(4,4))
    names = ['Loss', 'Accuracy']
    values = [loss, acc]
    colors = ['tab:red', 'tab:green']
    plt.bar(names, values, color=colors)
    plt.ylim(0, max(1.0, max(values) * 1.1))
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.title('Test Loss and Accuracy')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_roc_all(y_true_bin, y_score, class_names, out_path):
    n_classes = y_true_bin.shape[1]
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i], tpr[i], roc_auc[i] = np.array([0,1]), np.array([0,1]), 0.0

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr, dtype=float)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], color='k', linestyle='--', lw=2, label=f'Macro (AUC={roc_auc["macro"]:.3f})')
    plt.plot([0,1], [0,1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves'); plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    return roc_auc.get("macro", None)


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        fallback = Path('dataset/processed/test_clean.csv')
        csv_path = fallback if fallback.exists() else Path('dataset/processed/test.csv')
    print("Using test CSV:", csv_path)

    df = pd.read_csv(csv_path)
    label_col = detect_label_column(df)
    print("Detected label column:", label_col)

    paths = df['image_path'].astype(str).tolist()
    labels_raw = df[label_col].tolist()
    labels_mapped, mapping, inv_mapping = map_string_labels(labels_raw)
    class_names = [inv_mapping[i] for i in sorted(inv_mapping.keys(), key=lambda x: int(x))]

    print("Total CSV entries:", len(paths))
    ds = build_dataset(paths, labels_mapped, batch_size=args.batch_size, img_size=args.img_size)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print("Loading model:", model_path)
    model = load_model(str(model_path))
    print("Model loaded.")

    # Evaluate (get loss + metrics) using model.evaluate on dataset
    print("Running model.evaluate to compute loss/acc...")
    try:
        eval_res = model.evaluate(ds, verbose=1)
        # Keras returns [loss, metrics...], typical: [loss, accuracy]
        test_loss = eval_res[0] if isinstance(eval_res, (list, tuple)) else float(eval_res)
        test_acc = eval_res[1] if isinstance(eval_res, (list, tuple)) and len(eval_res) > 1 else None
    except Exception as e:
        print("model.evaluate failed (possibly due to dataset decoding). Falling back to predict for accuracy/loss placeholder.")
        test_loss = None
        test_acc = None

    # Predictions and gathering true labels
    print("Predicting probabilities...")
    y_scores = model.predict(ds, verbose=1)
    y_true_list = []
    for batch in ds.unbatch():
        y_true_list.append(int(batch[1].numpy()))
    y_true = np.array(y_true_list, dtype=int)

    if y_scores.ndim == 1:
        y_scores = np.vstack([1 - y_scores, y_scores]).T
    y_pred = np.argmax(y_scores, axis=1)

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        print("Length mismatch between true and pred; truncating to", min_len)
        y_true = y_true[:min_len]; y_pred = y_pred[:min_len]; y_scores = y_scores[:min_len]

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    if test_acc is None:
        test_acc = acc

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(y_scores.shape[1])), zero_division=0)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC macro
    y_true_bin = label_binarize(y_true, classes=list(range(y_scores.shape[1])))
    try:
        macro_auc = roc_auc_score(y_true_bin, y_scores, average='macro', multi_class='ovr')
    except Exception:
        macro_auc = None

    # Save textual report
    report_path = out_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Evaluation Report\n=================\n\n")
        f.write(f"Test CSV: {csv_path}\nModel: {model_path}\n\n")
        f.write(f"Evaluated samples: {len(y_true)}\n")
        if test_loss is not None:
            f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        if macro_auc is not None:
            f.write(f"Macro AUC (ovr): {macro_auc:.6f}\n")
        f.write("\nClassification Report:\n")
        f.write(clf_report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print("Saved textual report to", report_path)

    # Save plots
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    print("Saved confusion matrix.")

    plot_prf((precision, recall, f1), class_names, out_dir / "prf_per_class.png")
    print("Saved precision/recall/f1 chart.")

    macro_auc_plot = plot_roc_all(y_true_bin, y_scores, class_names, out_dir / "roc_curves.png")
    print("Saved ROC curves.")

    plot_loss_accuracy(test_loss if test_loss is not None else 0.0, test_acc if test_acc is not None else 0.0, out_dir / "loss_accuracy.png")
    print("Saved loss & accuracy plot.")

    # Print summary
    print("\n==== Summary ====")
    print("Samples:", len(y_true))
    print("Accuracy:", acc)
    if test_loss is not None:
        print("Test loss:", test_loss)
    if macro_auc is not None:
        print("Macro AUC:", macro_auc)
    print("\nClassification report:\n", clf_report)
    print("Confusion matrix:\n", cm)
    print("All outputs saved in", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/final_model.h5")
    parser.add_argument('--csv', type=str, default="dataset/processed/test.csv")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=260)
    parser.add_argument('--output_dir', type=str, default="results")
    args = parser.parse_args()
    main(args)
