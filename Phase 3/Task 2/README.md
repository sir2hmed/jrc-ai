# Phase 3 – Task 2: Deep Learning Fundamentals

## Overview

This repository contains my **Phase 3 – Task 2** presentation on fundamental deep learning concepts using **PyTorch**.

The presentation introduces the main ideas behind **Multi-Layer Perceptrons (MLPs)** and explains how a neural network is trained, regularized, evaluated, saved, and reloaded.

The presentation contains **28 slides** and is designed to explain the concepts in a simple and beginner-friendly way.

## Task Requirements

The task was to prepare a presentation covering:

- Multi-Layer Perceptrons (MLP)
- Forward propagation
- Backpropagation
- Weight initialization
- Loss functions
- Activation functions
- Optimizers
- Dropout
- Batch Normalization
- Learning rate scheduling
- Early Stopping
- Model checkpointing
- Model evaluation

## Topics Covered

### 1. Multi-Layer Perceptron
- Input, hidden, and output layers
- Neurons, weights, and biases
- `nn.Module`
- `nn.Linear`
- Basic MLP architecture in PyTorch

### 2. Forward & Backward Propagation
- Forward pass for generating predictions
- Loss calculation
- Backpropagation using PyTorch Autograd
- `loss.backward()`
- Updating weights using `optimizer.step()`

### 3. Weight Initialization
- Zero initialization
- Random initialization
- Xavier / Glorot initialization
- He / Kaiming initialization
- Orthogonal initialization

The presentation also explains why **Kaiming initialization** is commonly used with ReLU-based networks and **Xavier initialization** is commonly associated with Tanh and Sigmoid.

### 4. Loss Functions
Examples include:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Smooth L1 Loss
- Binary Cross-Entropy
- Cross-Entropy Loss

It also explains why `nn.CrossEntropyLoss()` expects **raw logits**.

### 5. Activation Functions
- ReLU
- LeakyReLU
- Sigmoid
- Tanh
- Softmax
- GELU

The presentation explains how activation functions introduce **non-linearity** into neural networks.

### 6. Optimizers
- SGD
- SGD with Momentum
- RMSProp
- Adam
- AdamW

Optimizers use gradients from backpropagation to update model parameters and reduce loss.

### 7. Dropout & Batch Normalization
The presentation explains:
- How Dropout helps reduce overfitting
- How Batch Normalization stabilizes training
- The difference between training and evaluation behavior

A common MLP block is shown as:

```text
Linear → BatchNorm → ReLU → Dropout
```

### 8. Learning Rate Scheduling & Early Stopping
Topics include:
- Learning rate control
- `ReduceLROnPlateau`
- Reducing the learning rate when validation loss stops improving
- Early Stopping
- Patience
- Preventing unnecessary training and overfitting

### 9. Learning Curves
The presentation includes examples of:
- Training vs validation loss
- Training vs validation accuracy
- Overfitting
- Underfitting
- Generalization

### 10. Model Checkpointing
Checkpointing is explained using:

```python
torch.save()
torch.load()
model.state_dict()
```

The goal is to save the **best-performing model**, rather than only saving the final epoch.

### 11. Model Evaluation

For classification:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC where appropriate

For regression:
- MAE
- MSE
- RMSE
- R²

The presentation also explains:

```python
model.eval()
```

and:

```python
with torch.no_grad():
```

## End-to-End Training Workflow

```text
Data
  ↓
Preprocessing
  ↓
MLP
  ↓
Forward Pass
  ↓
Loss
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Validation
  ↓
Checkpoint
  ↓
Final Evaluation
```

## Repository Structure

```text
.
├── phase3task2.pptx
└── README.md
```

## Presentation

Open:

`phase3task2.pptx`

The presentation follows the **JRCAI / Joint Research Center for AI (SDAIA - KFUPM)** theme.

## Author

**Ahmed Almutairi**

Supervised by **Dr. Safwan AlSelwi**

---

**Phase 3 – Task 2 | Deep Learning Fundamentals with PyTorch**
