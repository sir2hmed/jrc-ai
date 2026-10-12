# Student Performance Classification: PyTorch Deep Learning (Simple MLP)

A multi-class classification project using PyTorch to predict secondary student academic letter grades (**A, B, C, D, F**) based on socio-demographic, behavioral, and academic features.

This project represents **Phase 3** of the Student Performance analysis pipeline, transitioning from classical machine learning algorithms to a deep learning Multi-Layer Perceptron (MLP) architecture, complete with custom DataLoaders, evaluation metrics, and comparative benchmarking.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Results](#evaluation--results)
- [Phase 2 (Classical ML) vs. Phase 3 (Deep Learning)](#phase-2-classical-ml-vs-phase-3-deep-learning)
- [Single Sample Inference](#single-sample-inference)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Key Takeaways](#key-takeaways)

---

## 📖 Overview

The goal of this project is to accurately classify a student's final academic grade into one of 5 target categories:
$$\text{Classes} = \{A, B, C, D, F\}$$

### Workflow Steps
1. **Data Loading & Preprocessing**: Drop identifiers, label-encode categorical columns, stratified train/test split, and feature standardization.
2. **PyTorch Tensor Pipelines**: Wrapping scaled inputs into `TensorDataset` and batched, shuffled `DataLoader` instances.
3. **MLP Architecture Design**: Building a feed-forward neural network using `torch.nn.Module`.
4. **Model Training**: Optimizing parameters with `nn.CrossEntropyLoss` and the Adam optimizer over 30 epochs.
5. **Evaluation**: Computing classification reports, weighted F1-scores, accuracy, and confusion matrices on an unseen test set.
6. **Classical ML Benchmarking**: Comparing the PyTorch MLP against Decision Trees, Random Forests, Logistic Regression, SVM, and KNN.

---

## 📊 Dataset & Preprocessing

- **Dataset File**: `student_performance_dataset.csv`
- **Total Samples**: 1,000 records
- **Total Features**: 10 input features (after dropping `student_id`)
- **Target Feature**: `final_grade` (5 distinct classes: `A`, `B`, `C`, `D`, `F`)

### Preprocessing Steps
- **Identifier Removal**: `student_id` dropped as it carries no predictive signal.
- **Categorical Feature Encoding**: Encoded using `LabelEncoder`:
  - `gender`
  - `parental_education`
  - `internet_access`
  - `extracurricular_activities`
  - `part_time_job`
- **Target Encoding**: `final_grade` mapped into integer classes $\{0, 1, 2, 3, 4\}$.
- **Stratified Split**: 80% Training ($800$ samples) / 20% Testing ($200$ samples) with `stratify=y` to preserve class distributions.
- **Feature Standardization**: `StandardScaler` fit strictly on training data ($X_{\text{train}}$) and applied to both train and test splits to prevent data leakage.
- **Tensor Conversion**: Features mapped to `torch.float32` and targets to `torch.long`.

---

## 🧠 Model Architecture

The deep learning model is implemented as a modular `SimpleStudentMLP` inheriting from `torch.nn.Module`:

```
Input Layer (10 Features)
       │
       ▼
Linear Layer (10 -> 64)
       │
       ▼
ReLU Activation
       │
       ▼
Linear Layer (64 -> 32)
       │
       ▼
ReLU Activation
       │
       ▼
Output Layer (32 -> 5) [Raw Logits]
```

### PyTorch Definition
```python
class SimpleStudentMLP(nn.Module):
    def __init__(self, input_dim=10, hidden1=64, hidden2=32, output_dim=5):
        super(SimpleStudentMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
```

---

## ⚙️ Training Pipeline

| Parameter | Configuration |
| :--- | :--- |
| **Loss Function** | Cross-Entropy Loss (`nn.CrossEntropyLoss`) |
| **Optimizer** | Adam (`torch.optim.Adam`) |
| **Learning Rate** | `0.003` |
| **Batch Size** | `32` (25 batches / epoch) |
| **Epochs** | `30` |
| **Random Seed** | `42` (ensures exact reproducibility) |
| **Hardware** | Automatic device detection (`CUDA` if available, else `CPU`) |

### Training Progression
- **Epoch 1**: Loss: `1.3948` | Accuracy: `44.00%`
- **Epoch 5**: Loss: `0.3457` | Accuracy: `89.00%`
- **Epoch 10**: Loss: `0.1384` | Accuracy: `97.12%`
- **Epoch 20**: Loss: `0.0450` | Accuracy: `99.62%`
- **Epoch 30**: Loss: `0.0189` | Accuracy: `100.00%`

---

## 📈 Evaluation & Results

The model was evaluated on the unseen test split ($200$ samples):

- **Test Accuracy**: `91.00%`
- **Weighted F1-Score**: `0.9052`

### Classification Report

| Grade | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **A** | 0.92 | 0.98 | 0.95 | 57 |
| **B** | 0.90 | 0.92 | 0.91 | 71 |
| **C** | 0.96 | 0.85 | 0.90 | 52 |
| **D** | 0.81 | 0.94 | 0.87 | 18 |
| **F** | 0.00 | 0.00 | 0.00 | 2 |
| **Accuracy** | | | **0.91** | **200** |
| **Macro Avg** | 0.72 | 0.74 | 0.73 | 200 |
| **Weighted Avg** | 0.90 | 0.91 | 0.91 | 200 |

> **Note on Class Imbalance**: Grade `F` had only 2 support samples in the test set, reflecting natural real-world grade skewness. While high performance is achieved across grades `A` through `D`, minority-class techniques (e.g., class-weighted loss, oversampling) could further improve minority detection.

---

## 🔬 Phase 2 (Classical ML) vs. Phase 3 (Deep Learning)

All models were evaluated on the exact same $80/20$ stratified split and standardized inputs:

| Rank | Model | Paradigm | Accuracy (%) | Weighted F1-Score |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **Decision Tree** | Machine Learning | **99.50%** | **0.9954** |
| **2** | **Random Forest** | Machine Learning | **99.00%** | **0.9853** |
| **3** | **Logistic Regression** | Machine Learning | **94.50%** | **0.9400** |
| **4** | **PyTorch MLP** | Deep Learning | **91.00%** | **0.9052** |
| **5** | **Support Vector Machine (SVM)** | Machine Learning | **88.00%** | **0.8759** |
| **6** | **K-Nearest Neighbors (KNN)** | Machine Learning | **77.00%** | **0.7647** |

### Benchmark Insight
- Tree-based ensemble and rule models (**Decision Tree** and **Random Forest**) outperformed neural networks on this dataset. This aligns with empirical machine learning literature: tabular datasets with structured categorical/numerical boundaries and sample sizes under $\sim 10,000$ often favor gradient-boosted or decision-tree algorithms over deep networks, though the MLP demonstrates strong generalizability at **91.00%**.

---

## 🎯 Single Sample Inference

The notebook includes a live inference snippet demonstrating class-probability extraction via `torch.softmax`:

```python
model.eval()
with torch.no_grad():
    logits = model(sample_input)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_grade = class_names[pred_idx]
```

```text
Actual Grade    : A
Predicted Grade : A
Match           : ✓ CORRECT

Class Probabilities:
  Grade A: 100.00% █████████████████████████
  Grade B:   0.00% 
  Grade C:   0.00% 
  Grade D:   0.00% 
  Grade F:   0.00% 
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/student-performance-pytorch.git
cd student-performance-pytorch
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib
```

### 4. Launch the Notebook
```bash
jupyter notebook "Phase 3 Proj.ipynb"
```

---

## 📂 Project Structure

```text
.
├── Phase 3 Proj.ipynb            # Main Jupyter notebook containing training & evaluation
├── student_performance_dataset.csv # Dataset (1000 samples, 12 columns)
└── README.md                     # Project documentation
```

---

## 💡 Key Takeaways

1. **End-to-End PyTorch Workflow**: Successfully implemented a clean training loop featuring batching, gradient zeroing, forward pass, backpropagation, and loss tracking.
2. **Convergence**: Reached near-zero cross-entropy loss within 30 epochs with Adam optimization.
3. **Tabular Modeling Lessons**: Demonstrates why benchmarking deep learning against classical tabular models (Decision Trees, Random Forests) is essential before deploying deep architectures in production.
