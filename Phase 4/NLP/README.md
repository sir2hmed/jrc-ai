# 🎬 Natural Language Processing (NLP) Using PyTorch
> **End-to-End Sentiment Analysis on the IMDb Movie Reviews Dataset**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Project Overview

This repository contains a comprehensive, end-to-end Natural Language Processing (NLP) pipeline built in **PyTorch** and **Scikit-Learn**. The project explores text classification paradigms ranging from traditional n-gram vectorization methods to deep neural networks (RNN, LSTM, GRU) and modern **Transformer Encoders**.

All architectures are trained and benchmarked on the classic **IMDb Movie Reviews Dataset** for binary sentiment classification (`0` = Negative, `1` = Positive).

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Summary](#-dataset-summary)
- [Model Performance Benchmark](#-model-performance-benchmark)
- [Architectures Evaluated](#-architectures-evaluated)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Interactive Inference](#-interactive-inference)
- [Key Theoretical Insights](#-key-theoretical-insights)
- [License](#-license)

---

## ✨ Key Features

- **Text Preprocessing Pipeline:** HTML tag stripping, punctuation removal, case folding, custom tokenization, and out-of-vocabulary (`<unk>`, `<pad>`) handling.
- **Vocabulary Engineering:** Custom 10,000-word vocabulary builder mapped to integer index lookups.
- **Representation Comparisons:** Bag-of-Words (BoW), TF-IDF, and Dense Trainable Lookup Embeddings (`nn.Embedding`).
- **6 Model Architectures:** Evaluated under identical hardware, train/val/test splits, and evaluation criteria.
- **Interactive Classifier:** Custom `predict_sentiment()` function with visual confidence meters for test reviews.

---

## 📊 Dataset Summary

- **Source:** Hugging Face `datasets` (`imdb`)
- **Task:** Binary Sentiment Classification
- **Class Distribution:** Exactly **50% Positive / 50% Negative** across all splits.
- **Sequence Length Strategy:** Median length is 173 words. A sequence cutoff of `max_len = 200` was established to optimize computation while capturing core review sentiment.

---

## 🚀 Model Performance Benchmark

All models were evaluated on an identical test split under standard metrics.

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Parameters | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoW + Logistic Regression** | 84.65% | 85.61% | 83.30% | 84.44% | 5,001 | ~1.17s |
| **TF-IDF + Logistic Regression** | **86.65%** | **86.98%** | **86.20%** | **86.59%** | 5,001 | ~2.31s |
| **PyTorch Simple RNN** | 76.60% | 81.74% | 68.50% | 74.54% | 1,029,557 | ~27.48s |
| **PyTorch LSTM** | 79.25% | 79.94% | 78.10% | 79.01% | 1,118,293 | ~15.19s |
| **PyTorch GRU** | 77.70% | 75.60% | 81.80% | 78.58% | 1,088,725 | ~35.09s |
| **PyTorch Transformer Encoder** | 79.05% | 77.59% | 81.70% | **79.59%** | **706,497** | ~49.93s |

*Note: Among deep learning models trained from scratch without pre-trained weights, the Transformer Encoder achieved the highest F1-Score (79.59%) while using ~37% fewer parameters than LSTM.*

---

## 🏗 Architectures Evaluated

### 1. Traditional Baselines
- **BoW & TF-IDF + Logistic Regression:** High-performance linear baselines capturing strong sentiment keywords (*"masterpiece"*, *"terrible"*, *"waste"*).

### 2. Recurrent Architectures
- **Simple RNN:** Evaluates sequential processing; highlights the **vanishing gradient problem** across long time steps ($T = 200$).
- **LSTM (Long Short-Term Memory):** Uses a linear **Cell State ($C_t$)** and 3 gating mechanisms (*Forget, Input, Output*) to preserve long-range dependencies.
- **GRU (Gated Recurrent Unit):** Streamlined gating mechanism (*Reset, Update*) operating with fewer parameters.

### 3. Transformer Encoder
- **Multi-Head Self-Attention:** Eliminates sequential recurrence bottlenecks ($O(1)$ path length between any two tokens).
- **Scaled Dot-Product Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Positional Encoding:** Sinusoidal embeddings to inject sequence order awareness.

---

## 📁 Repository Structure

```text
.
├── nlp_pytorch_sentiment_analysis.ipynb   # Complete interactive Jupyter Notebook
├── README.md                              # Repository documentation
└── requirements.txt                       # Dependency specifications
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/nlp-pytorch-sentiment-analysis.git
   cd nlp-pytorch-sentiment-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio datasets scikit-learn nltk pandas numpy matplotlib seaborn
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook
   ```

---

## 💻 Interactive Inference

You can run custom predictions directly in Python using the trained models:

```python
# Example inference execution
review = "A visually stunning experience with a surprisingly deep, thoughtful, and emotional storyline."
result = predict_sentiment(review, model=transformer_model)

print(f"Sentiment:  {result['Sentiment']}")
print(f"Confidence: {result['Confidence']}")
print(f"Meter:      {result['Confidence_Meter']}")
```

**Output:**
```text
Sentiment:  Positive
Confidence: 99.4%
Meter:      [###################-]
```

---

## 💡 Key Theoretical Insights

1. **Why TF-IDF Performed Best:** IMDb sentiment analysis is strongly driven by explicit sentiment keywords. Linear models optimizing convex cross-entropy over n-gram TF-IDF representations excel rapidly without requiring huge corpora.
2. **Gradient Vanishing in RNNs:** Standard RNNs struggle over $T=200$ sequences due to repeated weight matrix multiplications during Backpropagation Through Time (BPTT).
3. **Gating & Attention Benefits:** LSTMs and Transformers effectively mitigate gradient decay, outperforming Simple RNNs across accuracy and F1 metrics.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
