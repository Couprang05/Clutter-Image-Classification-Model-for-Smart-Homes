
---

#  **Clutter Image Classification for Smart Homes**

*A Deep Learning–based Model using EfficientNetB2*

---
##  **Abstract**

Clutter detection is an emerging requirement in smart home automation, assistive robotics, and environment-aware AI systems. High levels of indoor clutter can obstruct robot navigation, reduce task efficiency, and increase safety risks for elderly or dependent individuals. This project proposes a deep learning–driven framework for automated clutter-level classification using indoor scene images. The system categorizes images into low, medium, and high clutter levels, enabling intelligent devices to assess environmental complexity and respond accordingly.

The model is trained using the MIT Indoor Scenes Dataset, where 67 scene categories are manually mapped to clutter labels. A complete preprocessing pipeline was developed, including dataset standardization, corrupted image removal, stratified train–validation–test splitting, and on-the-fly data augmentation. Pretrained EfficientNetB2, known for its compound scaling properties and superior efficiency, is adopted as the backbone. Using transfer learning, the model initially trains with frozen ImageNet weights to stabilize convergence. This is followed by fine-tuning of the deeper convolutional layers, allowing the network to adapt high-level visual features to clutter-specific spatial distributions and object densities.

The training protocol incorporates class weights to mitigate label imbalance and Adam optimization for stable gradient updates. Performance is evaluated using accuracy, classification reports, F1-scores, and confusion matrices on a held-out test set. Experimental results demonstrate that the fine-tuned EfficientNetB2 model reliably distinguishes varying clutter levels despite high intra-class variability. This indicates strong applicability in real-world scenarios such as autonomous cleaning robots, smart home monitoring, eldercare assistance, and indoor safety assessment.

Overall, the project establishes an effective and computationally efficient pipeline for clutter recognition, showcasing the potential of modern deep learning architectures in enabling context-aware smart environments.

---

##  **Project Overview**

Modern smart homes require systems that can understand and interpret the environment to assist with tasks like cleaning, organizing, and monitoring safety. One important aspect of this understanding is recognizing **clutter levels** in indoor environments.

This project uses **deep learning** to classify indoor images into **low**, **medium**, and **high** clutter levels using the MIT Indoor Scenes Dataset combined with a custom clutter labeling strategy.

The final model is built using **EfficientNetB2** with **transfer learning** and **fine-tuning**, trained entirely from scratch using your custom pipeline.

---

##  **Objectives**

* Build a reliable deep-learning classifier for clutter-level prediction
* Apply data preprocessing, augmentation, and stratified splitting
* Use transfer learning with EfficientNetB2
* Train, fine-tune, and evaluate the model on indoor scene images
* Analyze performance with metrics and confusion matrix
* Demonstrate deployment-ready classification pipeline

---

#  **Dataset**

###  **Dataset Source: MIT Indoor Scenes Dataset**

[https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)

The dataset contains **67 categories** of indoor scene images (total ~15K images). Examples include kitchens, libraries, meeting rooms, laundromats, gymnasiums, etc.

###  **Why This Dataset?**

* Diverse indoor environments
* Rich visual variety (textures, objects, layouts)
* Ideal for clutter-level inference
* Good real-world applicability for smart homes

---

##  **Custom Clutter Label Mapping**

Since clutter level isn't provided directly, each scene category was manually mapped into:

### **High Clutter**

Environments packed with objects, visual noise, or high density (e.g., bookstores, toy stores, kitchens, garages).

 *High clutter environments challenge object detection & navigation.*

### **Medium Clutter**

Places that contain moderate objects or organized density (e.g., classrooms, corridors, auditoriums).

 *Balanced layouts with moderate distraction.*

### **Low Clutter**

Minimal objects; visually open spaces (e.g., bathrooms, churches, staircases).

 *Clean, simple environments with fewer visual elements.*

---

#  **Data Preprocessing Pipeline**

This project includes a complete preprocessing workflow:

### **1. Folder-to-CSV Conversion**

All image paths and clutter labels are extracted and saved in
`mit_indoor_clutter_dataset.csv`.

 *Makes dataset easy to manage, inspect, and split.*

### **2. Cleaning Invalid Images**

Script checks for corrupted/truncated/unreadable files using PIL.

 *Ensures training and testing runs smoothly without decode errors.*

### **3. Stratified Train/Val/Test Split**

* **Train:** 70%
* **Validation:** 15%
* **Test:** 15%

 *Keeps clutter-level distribution equal across all splits.*

### **4. Preprocessing Steps**

* Resizing to **260x260**
* Normalizing pixel values (scaling 0–1)
* Ensuring all images are **RGB (3 channels)**

 *Standardizes the input for EfficientNet.*

### **5. Data Augmentation (Training Only)**

* Random horizontal flips
* Zoom
* Rotation
* Brightness and contrast changes

 *Improves generalization, reduces overfitting.*

---

#  **Model Architecture**

###  **EfficientNetB2 (Pretrained on ImageNet)**

This is a state-of-the-art model optimized using:

* **Compound scaling**
* Balanced width/depth/resolution
* High accuracy at low computation cost

### **Why EfficientNetB2?**

* Better accuracy than ResNet/VGG for same compute
* Performs exceptionally well on clutter & indoor scenes
* Lightweight enough to train on CPU
* Pretrained ImageNet weights accelerate convergence

 *Perfect balance of performance and efficiency.*

---

#  **Training Strategy**

### **1. Stage 1 — Frozen Backbone Training**

* Freeze EfficientNetB2 base
* Train classification head only
* Stabilizes weights
* Prevents catastrophic forgetting
* Learn dataset-specific patterns with minimal risk

 *Acts as a warm-up.*

### **2. Stage 2 — Fine-Tuning**

* Unfreeze last **40 layers**
* Train with **very low LR (1e-5)**
* Allows model to adapt deeper features
* Improves accuracy & feature extraction

 *Significantly boosts performance.*

### **Optimization Details**

* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Balanced Training: **Class Weights** applied to handle imbalance

---

#  **Evaluation Metrics**

### Included Metrics:

* **Accuracy** – overall correctness
* **Precision** – clutter predictions quality
* **Recall (Sensitivity)** – ability to capture cluttered scenes
* **Specificity** – ability to detect low clutter
* **F1-score** – harmonic balance of precision + recall
* **Confusion Matrix** – class-wise breakdown
* **Classification Report** – detailed summary

 *Gives complete insight into strengths/weaknesses of model.*

---

#  **Theory Behind Concepts Used**

### **1. CNNs (Convolutional Neural Networks)**

CNNs learn spatial hierarchies of features:

* Edges → shapes → textures → objects
* Ideal for clutter detection, where spatial relationships matter
* Automatically learn features (no manual engineering required)

### **2. Transfer Learning**

Using pretrained weights from ImageNet:

* Speeds up training
* Requires less data
* Improves generalization
* Leverages rich feature representations

### **3. EfficientNet Architecture**

EfficientNet scales model dimensions using:

* Depth (layers)
* Width (channels)
* Resolution (input size)

Unlike older architectures that scale only one dimension, EfficientNet improves accuracy **without huge computational cost.**

### **4. Fine-Tuning**

Gradually unfreezing layers:

* Allows deeper layers to adapt
* Avoids overfitting
* Achieves optimal feature learning for domain-specific tasks (clutter recognition)

---

#  **Project Structure**

```
Clutter_Image_Classification/
│
├── dataset/
│   ├── raw/                  # Original MIT scene images
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── test_clean.csv
│       └── bad_files.csv
│
├── models/                   # Saved checkpoints
│   ├── best_frozen.h5
│   ├── best_finetuned.h5
│   └── final_model.h5
│
├── py_scripts/
│   ├── data_pipeline.py      # preprocessing + loading
│   ├── model_EfficientNetB2.py
│   ├── mod_train.py          # training script
│   ├── evaluate_saved_model.py
│   └── check_test_images.py
│
└── README.md
```

---

#  **Installation & Setup**

### **1. Clone Repo**

```
git clone https://github.com/<username>/<repo>
cd <repo>
```

### **2. Install Dependencies**

```
pip install -r requirements.txt
```

---

#  **How to Run**

### **1. Preprocess Data**

```
python py_scripts/dataset_clean_split.py
```

### **2. Train Model**

```
python py_scripts/mod_train.py
```

### **3. Evaluate Model**

```
python py_scripts/evaluate_saved_model.py
```

---

#  **Future Improvements**

* Use **EfficientNetV2** for higher accuracy
* Try **Vision Transformers (ViT)**
* Add **Grad-CAM visualizations**
* Deploy model via Flask/Streamlit
* Use TFLite/ONNX for mobile integration
* Build a real-time smart home clutter monitor



#  **Conclusion**

This project demonstrates:

* Complete deep learning workflow
* Transfer learning with EfficientNetB2
* Effective clutter-level classification
* Thorough preprocessing + evaluation pipeline

It provides a strong foundation for smart home applications including:

* Autonomous cleaning
* Object organization
* Home robots
* Assistive living systems
