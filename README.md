# INT305-MNIST-CNN-Analysis
PyTorch-based CNN for MNIST digit classification, comparing performance with MobileNet and Vision Transformer (ViT). Includes model training, evaluation, and optimization techniques to enhance accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results and Discussion](#results-and-discussion)

---

## Introduction
This project explores different deep learning architectures for classifying handwritten digits from the **MNIST dataset**. We implement and compare the following models:
- **Basic CNN**: A simple convolutional neural network.
- **MobileNet**: A lightweight CNN optimized for efficiency.
- **Vision Transformer (ViT)**: A Transformer-based model for image classification.
- **EfficientFormer**: A hybrid model designed for efficient vision processing.

The goal is to evaluate how these models perform in terms of accuracy, computational efficiency, and generalization.

---

## Dataset Description
The **MNIST dataset** consists of **60,000 training images** and **10,000 test images** of handwritten digits (0-9), with each image being **28x28 grayscale pixels**.

### Data Preprocessing
- **CNN & MobileNet**:
  - Images are normalized to the range **[-1,1]**.
  - No resizing required (**28x28 input size**).

- **ViT & EfficientFormer**:
  - Images are **resized to 224x224** for transformer models.
  - **Grayscale converted to 3 channels** for EfficientFormer.

---

## Methodology

### 1. **Model Architectures**
#### **Basic CNN**
- **2 convolutional layers** for feature extraction.
- **Max pooling layers** for dimensionality reduction.
- **Fully connected layers** for classification.

#### **MobileNet**
- Uses **Depthwise Separable Convolutions** to reduce computation.
- Global average pooling before the classification layer.

#### **Vision Transformer (ViT)**
- Splits the image into **patches** and processes them using **self-attention**.
- Uses **position embeddings** to retain spatial information.

#### **EfficientFormer**
- A lightweight Transformer-based architecture optimized for speed and accuracy.
- Uses an efficient attention mechanism with **feed-forward layers**.

### 2. **Training Process**
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.001)
- **Batch Size**: 64
- **Epochs**:
  - CNN: 3 epochs
  - MobileNet: 5 epochs
  - ViT: 10 epochs
  - EfficientFormer: 5 epochs

- **Evaluation Metrics**:
  - **Accuracy**
  - **Loss Curve**
  - **Confusion Matrix**

---

## Results and Discussion

| Model         | Training Accuracy | Test Accuracy | Training Time | Key Features |
|--------------|------------------|--------------|---------------|--------------|
| **Basic CNN**  | 99.47%           | 98.95%       | **Fastest** (~3 min) | Simple CNN with 2 conv layers |
| **MobileNet** | 99.38%           | 98.88%       | **Efficient** (~5 min) | Uses depthwise separable convolutions |
| **ViT**       | 96.95%           | 98.90%       | **Slowest** (~15 min) | Uses self-attention for feature extraction |
| **EfficientFormer** | 99.30%    | 98.91%       | **Balanced** (~8 min) | Optimized transformer with reduced computational overhead |

### **Key Observations**
- **CNN and MobileNet** performed **slightly better** than ViT on MNIST, likely due to the small dataset size.
- **MobileNet** had the **best efficiency** (fastest with minimal accuracy loss).
- **ViT required more epochs** to reach comparable accuracy but provides strong feature extraction capabilities.
- **EfficientFormer achieved a balance** between accuracy and computational efficiency.

---

## Future Improvements
- **Train on larger datasets** (e.g., CIFAR-10, ImageNet) to better evaluate **ViT** and **EfficientFormer**.
- **Use data augmentation** (e.g., rotation, scaling) to enhance model generalization.
- **Experiment with hyperparameter tuning**, such as:
  - Learning rate schedules (e.g., ReduceLROnPlateau).
  - Different optimizers (e.g., AdamW, SGD).
- **Enhance interpretability** using **SHAP** or **Grad-CAM**.
