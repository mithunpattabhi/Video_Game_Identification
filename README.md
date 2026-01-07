# 🎮 Video Game Screenshot Classification using EfficientNet

This project is a deep learning–based image classification system that identifies **which video game a screenshot belongs to**, using a dataset built completely from scratch.

The model classifies screenshots from the following games:

- GTA V  
- Indiana Jones  
- Marvel’s Spider-Man  
- Tomb Raider  

This project covers the **entire machine learning pipeline** — data collection, preprocessing, training, debugging, evaluation, and deployment.

---

## Motivation

Most ML projects use clean, prebuilt datasets.  
This project was intentionally built **without any existing dataset** to experience real-world ML challenges.

The goals were to:
- Build a dataset from gameplay videos(youtube)
- Train a robust image classifier on visually similar games
- Debug model collapse and bias issues
- Deploy the final model with a simple UI

---

## Dataset Creation

There was no public dataset available, so everything was created manually.

### 1️ Gameplay Video Collection
- Gameplay videos were sourced from YouTube
- Multiple videos per game were used
- Different scenes, lighting, environments, and HUDs were included

### 2️ Frame Extraction
- Frames were extracted from videos at regular intervals using OpenCV
- Low-quality and duplicate frames were avoided
- Frames were stored in class-wise directories

### 3️ Dataset Balancing
- Initially, some classes had far more frames than others
- Excess frames were reduced to prevent class bias
- Final dataset was balanced across all classes

---

## Model Architecture

- **Base Model:** EfficientNetB0 (pretrained on ImageNet)
- **Custom Head:**
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Softmax output layer

EfficientNet was chosen for its strong performance on limited datasets and efficient parameter usage.

---

## Training Pipeline

- Image size: **224 × 224**
- Pixel normalization: **[0, 1]**
- Data augmentation:
  - Horizontal flip
  - Small rotations
  - Zoom and contrast variations

### Training Strategy
1. Train classification head with backbone frozen
2. Fine-tune deeper layers with a lower learning rate

---

## Challenges Faced & Solutions

### Class Collapse
At multiple stages, the model predicted only **one class for all inputs**.

**Causes identified:**
- Dataset imbalance
- Incorrect freezing strategy
- Over-augmentation
- Loss configuration issues

**Solutions:**
- Balanced dataset
- Gradual fine-tuning
- Reduced augmentation strength
- Correct loss usage

---

### Misleading Accuracy
Validation accuracy looked reasonable, but confusion matrix showed poor predictions.

**Fix:**
- Relied on confusion matrix, precision, recall, and F1-score instead of accuracy alone

---

### Deployment Bug (Streamlit)
Initially, the Streamlit app showed:
- Same prediction for all images
- Identical probability distributions

**Root cause:**
- Class index mismatch between training and inference

**Fix:**
- Saved and reused `class_names.json`
- Matched preprocessing exactly with training pipeline

---

## Final Results

- **Test Accuracy:** ~84%
- Balanced precision and recall across all classes
- No class collapse
- Strong generalization to unseen screenshots

The final confusion matrix shows clear diagonal dominance with minimal misclassification.
<img width="1008" height="809" alt="confusion_matrix" src="https://github.com/user-attachments/assets/3af38ddf-8501-41a2-9aaa-6c0f6e459199" />


---

## Streamlit Web App

A simple Streamlit UI was built to:
- Upload a game screenshot
- Display predicted game
- Show probability distribution across all classes

The app uses the final trained EfficientNet model with correct preprocessing.

---

## Tech Stack

- Python
- TensorFlow / Keras
- EfficientNet (Transfer Learning)
- OpenCV
- NumPy, Pandas
- Streamlit
- Google Colab

---

## Key Learnings

- Real-world ML involves extensive debugging
- Dataset quality matters more than model size
- Accuracy alone can be misleading
- Deployment issues can break an otherwise good model
- Transfer learning requires careful fine-tuning

---

## Future Improvements

- Add more games
- Use video-based models instead of single frames
- Experiment with temporal models (CNN + LSTM)
