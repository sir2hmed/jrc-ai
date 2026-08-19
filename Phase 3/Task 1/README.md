# PyTorch Fundamentals

A beginner-friendly introduction to the core concepts of **PyTorch**, built around one simple regression example:

> **Study Hours → Exam Score**

This presentation explains how PyTorch connects **data, tensors, neural networks, loss functions, autograd, optimizers, training loops, and inference** into one complete machine learning workflow.

## Presentation Information

- **Prepared by:** Ahmed Saleh Almutairi
- **Presented to:** Dr. Safwan Alsewi
- **Organization:** JRCAI — Joint Research Center for AI (SDAIA - KFUPM)

---

## Topics Covered

The presentation introduces the following PyTorch fundamentals:

1. What PyTorch is
2. A simple regression problem
3. Tensors
4. Tensor shapes, data types, and devices
5. CPU and GPU usage
6. Features and targets
7. `Dataset` and `DataLoader`
8. Building models with `nn.Module`
9. Forward propagation
10. Loss functions
11. Computational graphs
12. Automatic differentiation with Autograd
13. Optimizers and learning rates
14. The PyTorch training loop
15. Monitoring loss during training
16. Evaluation and inference
17. Complete end-to-end PyTorch workflow

---

## Example Problem

The presentation uses a small regression dataset:

| Study Hours | Exam Score |
|---:|---:|
| 1 | 45 |
| 2 | 52 |
| 3 | 60 |
| 4 | 68 |
| 5 | 76 |
| 6 | 84 |

The goal is to train a neural network that learns the relationship between the number of hours studied and the expected exam score.

For example:

```text
7 study hours
      ↓
Neural Network
      ↓
Predicted score ≈ 91.7
```

---

## PyTorch Workflow

```text
Data
  ↓
Tensor
  ↓
Model
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Weights
  ↓
Repeat
```

---

## Simple PyTorch Example

```python
import torch
import torch.nn as nn

# Training data
X = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
)

y = torch.tensor(
    [[45.0], [52.0], [60.0], [68.0], [76.0], [84.0]]
)

# Neural network
class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.network(x)


model = ScoreModel()

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Training
for epoch in range(2000):
    prediction = model(X)

    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()

with torch.no_grad():
    new_student = torch.tensor([[7.0]])
    predicted_score = model(new_student)

print(predicted_score.item())
```

---

## Core Concepts

### Tensors

A **tensor** is PyTorch's main structure for storing numerical data.

Examples:

```python
scalar = torch.tensor(5.0)

vector = torch.tensor([1.0, 2.0, 3.0])

matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
```

A scalar is a 0D tensor, a vector is a 1D tensor, and a matrix is a 2D tensor.

---

### `nn.Module`

PyTorch models are normally created by inheriting from:

```python
torch.nn.Module
```

The model's layers are defined inside `__init__()`, while `forward()` describes how data moves through the network.

---

### Forward Pass

The forward pass sends input data through the model:

```python
prediction = model(X)
```

This produces a prediction using the model's current parameters.

---

### Loss Function

A loss function measures how far the prediction is from the correct target.

For this regression example:

```python
criterion = nn.MSELoss()
```

Mean Squared Error:

1. Calculates the prediction error.
2. Squares the error.
3. Averages the errors across the samples.

Training attempts to reduce this loss.

---

### Computational Graph

During the forward pass, PyTorch records the mathematical operations that connect tensors and model parameters to the final loss.

This dependency graph allows PyTorch to calculate gradients automatically during backpropagation.

---

### Autograd

PyTorch's **Autograd** system automatically calculates gradients.

```python
loss.backward()
```

The gradients describe how changing each model parameter would affect the loss.

Autograd calculates the gradients, but it does **not** update the weights itself.

---

### Optimizer

The optimizer uses the calculated gradients to update the model parameters.

```python
optimizer.step()
```

This example uses Adam:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

The learning rate (`lr`) controls the size of parameter updates.

---

## Training Loop

A typical PyTorch training loop follows the same main steps:

```python
for epoch in range(num_epochs):

    # 1. Forward pass
    prediction = model(X)

    # 2. Calculate loss
    loss = criterion(prediction, y)

    # 3. Clear old gradients
    optimizer.zero_grad()

    # 4. Calculate new gradients
    loss.backward()

    # 5. Update parameters
    optimizer.step()
```

An **epoch** means one complete pass through the training dataset.

---

## Dataset and DataLoader

For very small datasets, tensors can be used directly.

For larger datasets, PyTorch provides:

### `Dataset`

Defines how individual samples are accessed.

### `DataLoader`

Handles tasks such as:

- Mini-batching
- Shuffling
- Iterating through the dataset
- Parallel data loading

Example:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

---

## CPU and GPU

PyTorch tensors and models can run on different hardware devices.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Then move the model and tensors:

```python
model = model.to(device)
X = X.to(device)
y = y.to(device)
```

GPUs are especially useful for large neural networks and large batches because many tensor operations can be executed in parallel.

---

## Evaluation and Inference

After training:

```python
model.eval()
```

This changes the behavior of layers such as Dropout and Batch Normalization.

For inference, gradient tracking can also be disabled:

```python
with torch.no_grad():
    prediction = model(new_data)
```

`model.eval()` and `torch.no_grad()` perform different jobs:

- `model.eval()` changes the behavior of specific model layers.
- `torch.no_grad()` disables gradient tracking.

---

## Main Takeaway

PyTorch training can be understood as one repeated learning cycle:

```text
Predict
  ↓
Measure the error
  ↓
Calculate gradients
  ↓
Update the weights
  ↓
Repeat
```

The main PyTorch components supporting this workflow are:

- **Tensor** — stores numerical data
- **`nn.Module`** — defines the model
- **Loss Function** — measures prediction error
- **Autograd** — calculates gradients
- **Optimizer** — updates parameters
- **Dataset / DataLoader** — organize and deliver data
- **Training Loop** — connects all components together
- **Inference** — uses the trained model to make new predictions

---

## Repository Purpose

This repository is intended as a simple learning resource for understanding the basic PyTorch workflow before moving to more advanced topics such as:

- Convolutional Neural Networks
- Recurrent Neural Networks
- Transformers
- Transfer Learning
- Mixed Precision Training
- Distributed Training
- Model Deployment

---

## Requirements

```bash
pip install torch
```

Check the installed version:

```python
import torch

print(torch.__version__)
```

---

## License

This material is intended for educational use.
