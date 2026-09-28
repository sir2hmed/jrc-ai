# Phase 3 – Task 2 | Deep Learning Fundamentals with PyTorch

**House Price Prediction using a Multi-Layer Perceptron (MLP)**

This project explores the fundamentals of deep learning through a **PyTorch regression model** trained on the King County house-sales dataset (`houseprice.csv`). It combines an educational presentation with a Jupyter Notebook that implements an MLP, compares training techniques, visualizes experimental results, and evaluates the trained model against classical machine-learning baselines.

**Author:** Ahmed Almutairi  
**Supervisor:** Dr. Safwan AlSelwi  
**Context:** Phase 3 – Task 2 | JRCAI (SDAIA–KFUPM Joint Research Center for AI)

## Project objectives

The assignment has two parts:

**Presentation (maximum 30 slides):** Explain MLPs, forward propagation, backpropagation, weight initialization, loss functions, activation functions, optimizers, Dropout, Batch Normalization, learning-rate scheduling, Early Stopping, checkpointing, and model evaluation.

**Implementation:** Build an MLP in PyTorch; compare activation functions and optimizers on the same dataset; experiment with Dropout and BatchNorm; generate training/validation learning curves; evaluate with regression metrics; and save and reload a trained model checkpoint.

## Dataset and preprocessing

- **Dataset:** `houseprice.csv` — King County / Washington house-sales data.
- **Target:** `price` (continuous USD value; regression).
- **Cleaning:** 49 records with a zero sale price are removed, leaving **4,551** rows in the recorded notebook run.
- **Leakage prevention:** `price_per_sqft` is excluded because it is derived from the target price; `street` is excluded because of its high cardinality.
- **Engineered features:** sale month/day, house age, renovation indicators, years since renovation, living-to-lot ratio, and bathroom-to-bedroom ratio.
- **Categorical encoding:** `city` and `statezip` are one-hot encoded; the recorded preprocessing produces **138 input features**.
- **Split:** 80% train (**3,640**), 10% validation (**455**), and 10% held-out test (**456**).
- **Scaling:** Feature and target `StandardScaler` objects are fitted on the training split; reported prediction errors are converted back to USD. The current notebook creates one-hot columns before splitting, so this part can be tightened in a future revision by fitting the encoder on training data only.

## MLP architecture

The configurable `HousePriceMLP(nn.Module)` uses the following final-run setup:

```text
138 processed input features
       ↓
Linear(138, 128) → BatchNorm → LeakyReLU(0.01) → Dropout(0.15)
       ↓
Linear(128, 64)  → BatchNorm → LeakyReLU(0.01) → Dropout(0.15)
       ↓
Linear(64, 32)   → BatchNorm → LeakyReLU(0.01) → Dropout(0.15)
       ↓
Linear(32, 1) → standardized predicted price
```

The last prediction is inverse-transformed to obtain the estimated price in USD. The final training configuration uses **MSELoss**, **Kaiming initialization**, **AdamW**, `ReduceLROnPlateau`, and Early Stopping.

## Experiments

| Experiment | Compared configurations |
|---|---|
| 1 — Weight initialization | Zero, Random Normal, Xavier Uniform, Kaiming Normal |
| 2 — Activation functions | Sigmoid, Tanh, ReLU, LeakyReLU |
| 3 — Optimizers | SGD with Momentum, RMSprop, Adam, AdamW |
| 4 — Regularization | No regularization, Dropout, BatchNorm, Dropout + BatchNorm |

The notebook then trains the selected final configuration, tracks training and validation loss, applies learning-rate scheduling and Early Stopping, saves the best weights, reloads them into a fresh model instance, and verifies that the reloaded model produces matching predictions.

## Recorded evaluation results

**These figures come from the saved output of the latest uploaded notebook (`ph3t2(1).ipynb`), not from a new independent rerun.** Every model in its benchmark is evaluated on the same held-out test split.

| Model | Test R² ↑ | Test MAE ↓ | Test RMSE ↓ |
|---|---:|---:|---:|
| **PyTorch MLP** | **0.7536** | **$106,803** | **$201,536** |
| Ridge Regression | 0.7070 | $124,309 | $219,764 |
| Gradient Boosting | 0.6752 | $121,573 | $231,374 |
| Random Forest | 0.6382 | $133,279 | $244,211 |

The notebook also benchmarks Decision Tree, SVR, and Linear Regression. Its saved output reports that the final run stopped at epoch **14**, restored the best weights from epoch **6**, and passed the checkpoint-reload comparison.

> **Results consistency:** An earlier presentation/experiment version reports different final MLP metrics (R² = 0.7829, MAE = $111,215, RMSE = $189,178). Before publishing the slides as the results of this notebook run, regenerate or reconcile those figures so the presentation and notebook describe the **same experiment**. Do not combine metrics from different runs.

## Repository structure

Recommended GitHub layout (rename the downloaded files if necessary):

```text
.
├── README.md
├── ph3t2.ipynb
├── Phase 3 Task 2 Pre.pptx
├── houseprice.csv                  # Add separately if redistribution is permitted
├── checkpoints/
│   └── best_houseprice_mlp.pth     # Generated by the notebook
└── presentation_graphs/           # Generated by the notebook
    ├── eda_target_and_view_distribution.png
    ├── experiment1_weight_initialization.png
    ├── experiment2_activation_functions.png
    ├── experiment3_optimizers.png
    ├── experiment4_regularization.png
    ├── final_model_learning_curves.png
    ├── prediction_vs_actual_scatter.png
    ├── benchmark_classical_vs_deep_learning.png
    └── presentation_summary_dashboard.png
```

If you keep the upload filenames `ph3t2(1).ipynb` and `Phase 3 Task 2 Pre(1).pptx`, replace the corresponding names above.

## How to run

**Requirements:** Python 3, Jupyter, PyTorch, pandas, NumPy, scikit-learn, Matplotlib, and Seaborn. PyTorch can run on CPU, CUDA, or Apple MPS if available.

1. Clone the repository and open its folder.
2. Install the dependencies:

   ```bash
   python -m pip install notebook ipykernel torch pandas numpy scikit-learn matplotlib seaborn
   ```

3. Place `houseprice.csv` in the same folder as `ph3t2.ipynb` (the notebook reads it using a relative path).
4. Launch Jupyter:

   ```bash
   jupyter notebook ph3t2.ipynb
   ```

5. Select your Python kernel and run the cells **from top to bottom**. The notebook creates `checkpoints/` and `presentation_graphs/` automatically.

A fixed random seed (`42`) is used, but exact training results can vary with platform, library versions, and hardware.

## Key PyTorch operations

```python
predictions = model(inputs)       # Forward propagation
loss = criterion(predictions, targets)
optimizer.zero_grad()            # Clear old gradients
loss.backward()                  # Backpropagation
optimizer.step()                 # Update model weights
```

During validation and final evaluation, the notebook calls `model.eval()` and uses `torch.no_grad()`. The saved checkpoint contains the **best model weights** (`state_dict`) rather than a full optimizer-resume package.

## Presentation

The PowerPoint presents each major concept with beginner-friendly explanations, practical PyTorch snippets, and experiment graphs. The practical example is **house-price regression**; classification losses such as Cross-Entropy are included as theory, not as the training objective for this model.

## Notes

- The CSV is required to rerun the project; verify its source and redistribution rights before committing it to a public repository.
- This is an **educational experiment**, not a production-ready real-estate valuation service.
- Use the notebook's recorded outputs as the source for any result tables or slide graphics, and keep the final presentation synchronized with the run being submitted.

---

**Phase 3 – Task 2 | Deep Learning Fundamentals with PyTorch**
