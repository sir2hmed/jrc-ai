# Phase 3 Task 2 — Deep Learning Fundamentals with PyTorch

## House Price Prediction Using a Multi-Layer Perceptron

This project applies deep learning fundamentals to a real-world house price regression problem using **PyTorch**. The main model is a configurable **Multi-Layer Perceptron (MLP)** trained on structured real-estate data .

The project focuses not only on obtaining predictions, but also on understanding the complete deep-learning workflow: preprocessing, forward propagation, loss calculation, backpropagation, weight initialization, activation functions, optimizers, regularization, learning-rate scheduling, early stopping, checkpointing, and final evaluation.

**Author:** Ahmed Almutairi  
**Supervisor:** Dr. Safwan AlSelwi  
**Organization:** JRCAI — Joint Research Center for AI (SDAIA–KFUPM)

---

## Project Objectives

The project demonstrates the following deep-learning concepts:

- Building a custom MLP using `torch.nn.Module`
- Forward propagation and backpropagation
- Mean Squared Error for regression
- Weight initialization techniques
- Activation-function comparison
- Optimizer comparison
- Dropout and Batch Normalization
- Learning-rate scheduling
- Early stopping
- Model checkpointing and reloading
- Regression evaluation using R², MAE, and RMSE
- Leakage-safe preprocessing and evaluation

---

## Dataset

The project uses:

`houseprice.csv`

**Dataset Source:** [House Sales Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/shree1992/housedata?resource=download)

The dataset contains real-estate sale information including property size, number of bedrooms and bathrooms, condition, view quality, location, renovation information, and sale price.

### Dataset Preview

The first five rows of the original dataset are shown below:

| date       |          price |   bedrooms |   bathrooms |   sqft_living |   sqft_lot |   floors |   waterfront |   view |   condition |   sqft_above |   sqft_basement |   yr_built |   yr_renovated | street                   | city      | statezip   |   price_per_sqft |
|:-----------|---------------:|-----------:|------------:|--------------:|-----------:|---------:|-------------:|-------:|------------:|-------------:|----------------:|-----------:|---------------:|:-------------------------|:----------|:-----------|-----------------:|
| 2014-05-02 | 313000         |          3 |        1.5  |          1340 |       7912 |      1.5 |            0 |      0 |           3 |         1340 |               0 |       1955 |           2005 | 18810 Densmore Ave N     | Shoreline | WA 98133   |           233.58 |
| 2014-05-02 |      2.384e+06 |          5 |        2.5  |          3650 |       9050 |      2   |            0 |      4 |           5 |         3370 |             280 |       1921 |              0 | 709 W Blaine St          | Seattle   | WA 98119   |           653.15 |
| 2014-05-02 | 342000         |          3 |        2    |          1930 |      11947 |      1   |            0 |      0 |           4 |         1930 |               0 |       1966 |              0 | 26206-26214 143rd Ave SE | Kent      | WA 98042   |           177.2  |
| 2014-05-02 | 420000         |          3 |        2.25 |          2000 |       8030 |      1   |            0 |      0 |           4 |         1000 |            1000 |       1963 |              0 | 857 170th Pl NE          | Bellevue  | WA 98008   |           210    |
| 2014-05-02 | 550000         |          4 |        2.5  |          1940 |      10500 |      1   |            0 |      0 |           4 |         1140 |             800 |       1976 |           1992 | 9105 170th Ave NE        | Redmond   | WA 98052   |           283.51 |

The prediction target is `price`.

`price`

This is a **regression problem** because the model predicts a continuous numerical value.

### Data Cleaning

Rows with non-positive house prices are removed before training.

`df = df_raw[df_raw["price"] > 0].copy()`

The original dataset contains approximately 4,600 records, and 4,551 valid sales remain after removing zero-price records.

---

## Feature Engineering

Several additional features are created from the original dataset:

`sale_month`, `sale_day`, `is_renovated`, `living_to_lot_ratio`, `bath_to_bed_ratio`

The original date column is converted to pandas datetime format so the sale month and day can be extracted.

`df["sale_date"] = pd.to_datetime(df["date"])`

This allows the sale month and day to be extracted.

A binary renovation indicator is created from `yr_renovated`.

`df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)`

Additional ratio features help the model describe relationships between property characteristics rather than relying only on raw values.

---

## Leakage Prevention

The project explicitly removes features that could produce misleading model performance.

### `price_per_sqft`

`price_per_sqft` is excluded because it is calculated directly from the target:

\[
\text{price\_per\_sqft} =
\frac{\text{price}}{\text{sqft\_living}}
\]

Using it to predict `price` would expose information about the answer to the model.

### `sqft_above`

`sqft_above` is also excluded because it is deterministically related to other available features:

\[
\text{sqft\_living}
=
\text{sqft\_above}
+
\text{sqft\_basement}
\]

Removing the redundant feature improves the stability of classical linear baselines.

---

## Train, Validation, and Test Split

The dataset is divided approximately into:

`80% Training`, `10% Validation`, `10% Test`

The project uses `GroupShuffleSplit` with `street` as the grouping key.

This prevents the same property address from appearing in multiple splits.

The `street` column is used only for grouping and is not provided to the model as a training feature.

---

## Categorical Encoding

The categorical features are `city` and `statezip`.

`city`, `statezip`

They are converted into numerical form using:

The categorical encoder uses `OneHotEncoder` with unseen-category handling and is fitted only on the training split.

The encoder is fitted **only on the training dataset**.

Validation and test data are transformed using the already-fitted encoder. This prevents information from the validation or test sets from influencing preprocessing.

`handle_unknown="ignore"` also allows the pipeline to process categories that were not observed during training.

---

## Feature Scaling

The project uses `StandardScaler` for both the input features and target variable.

Both the input features and target are standardized using `StandardScaler`, fitted only on the training data.

StandardScaler transforms features using:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original value
- \(\mu\) is the training-set mean
- \(\sigma\) is the training-set standard deviation

The scalers are fitted only on the training set:

`X_train = scaler_X.fit_transform(X_train_raw)`, `X_val = scaler_X.transform(X_val_raw)`, `X_test = scaler_X.transform(X_test_raw)`

Scaling helps neural-network optimization by keeping features with very different numerical ranges on comparable scales.

---

## PyTorch MLP Architecture

The neural network is implemented as a reusable custom PyTorch `nn.Module`.

`class HousePriceMLP(nn.Module):`

The default architecture is **Input → 128 → 64 → 32 → 1**.

`Input`, `↓`, `128 neurons`, `↓`, `64 neurons`, `↓`, `32 neurons`, `↓`, `1 output`

Or mathematically:

\[
\text{Input} \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1
\]

The number of input features is calculated dynamically after preprocessing rather than being hard-coded.

The final output contains one neuron because the model predicts one continuous value: the house price.

### Configurable Components

The model supports different:

- Activation functions
- Dropout rates
- Batch Normalization settings
- Hidden-layer dimensions
- Input feature counts

This allows controlled experiments without rewriting the neural-network architecture.

---

## Forward Propagation and Backpropagation

A standard PyTorch training step follows this process:

```python
optimizer.zero_grad()
predictions = model(x)
loss = criterion(predictions, y)
loss.backward()
optimizer.step()
```

### Forward Pass

`predictions = model(x)`

The features pass through the neural network and produce a prediction.

### Loss

`loss = criterion(predictions, y)`

The prediction is compared with the true value.

### Backpropagation

`loss.backward()`

PyTorch automatically calculates how each model parameter contributed to the prediction error.

### Weight Update

`optimizer.step()`

The optimizer uses the calculated gradients to update the model weights.

---

## Loss Function

The project uses:

`nn.MSELoss()`

Mean Squared Error is appropriate for regression because it measures the squared difference between predicted and actual values:

\[
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

Large prediction errors receive a stronger penalty because the errors are squared.

---

## Experiment 1 — Weight Initialization

The project compares four initialization strategies:

`Zero Initialization`, `Random Normal Initialization`, `Xavier Uniform Initialization`, `Kaiming / He Normal Initialization`

### Zero Initialization

All weights begin at zero.

This is generally unsuitable for hidden-layer weights because neurons begin identically and cannot learn diverse reanalysiss.

### Random Normal

Weights begin with small random values.

Random initialization breaks symmetry so different neurons can learn different patterns.

### Xavier / Glorot

Xavier initialization scales the initial weights based on the number of inputs and outputs.

It is commonly associated with activation functions such as Tanh and Sigmoid.

### Kaiming / He

Kaiming initialization is designed for ReLU-family activations.

The implementation also correctly supports LeakyReLU:

Kaiming initialization is configured to match ReLU-family activations, including LeakyReLU.

The initialization experiment is evaluated using validation performance rather than training loss alone.

---

## Experiment 2 — Activation Functions

The following activation functions are compared:

`Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`

### ReLU

\[
f(x)=\max(0,x)
\]

ReLU is computationally simple and widely used in hidden neural-network layers.

### LeakyReLU

LeakyReLU allows a small negative output instead of forcing every negative value to zero.

The project uses:

`nn.LeakyReLU(negative_slope=0.01)`

### Sigmoid

Sigmoid maps values between 0 and 1.

### Tanh

Tanh maps values between -1 and 1.

The activation selected for the final model is determined from the validation experiments rather than being manually declared in advance.

---

## Experiment 3 — Optimizers

The project compares:

`SGD with Nesterov Momentum`, `RMSprop`, `Adam`, `AdamW`

### SGD

Stochastic Gradient Descent directly updates parameters using gradients.

The project uses momentum and Nesterov acceleration:

SGD is evaluated with momentum `0.9` and Nesterov acceleration.

### RMSprop

RMSprop adapts updates based on recent gradient magnitudes.

### Adam

Adam combines momentum-like behavior with adaptive learning rates.

### AdamW

AdamW separates weight decay from the gradient update, providing a cleaner form of weight regularization.

A common learning rate is used in the controlled optimizer experiment so the optimizer itself remains the main changing variable.

---

## Experiment 4 — Dropout and Batch Normalization

The project compares four configurations:

`Baseline`, `Dropout`, `BatchNorm`, `BatchNorm + Dropout`

### Dropout

Dropout randomly disables some neuron outputs during training:

`nn.Dropout(p=0.2)`

Its purpose is to reduce overfitting and prevent the model from depending too strongly on specific neurons.

During evaluation, Dropout is automatically disabled when:

`model.eval()`

is used.

### Batch Normalization

Batch Normalization normalizes intermediate layer activations:

`nn.BatchNorm1d(hidden_dim)`

It can improve training stability and optimization.

A regularized block can therefore follow:

`Linear`, `↓`, `BatchNorm`, `↓`, `Activation`, `↓`, `Dropout`

The corrected experiment compares validation loss directly instead of using a misleading train-validation loss gap.

---

## Controlled Experimental Design

To improve reproducibility, each experimental configuration is evaluated using multiple random seeds:

`EXPERIMENT_SEEDS = [42, 52, 62]`

The experiments use:

- The same dataset splits
- The same seeded batch order
- The same architecture where appropriate
- The best validation epoch
- Validation data for model selection
- No test-set tuning

Results are summarized using the mean and standard deviation across runs.

This reduces the chance that a configuration appears superior only because of one favorable random initialization.

---

## Final Model Selection

The final model configuration is selected automatically from the validation experiments.

The notebook determines:

`Best initialization`, `Best activation`, `Best optimizer`, `Best regularization configuration`

The selected components are then combined into the final MLP.

This prevents the final architecture from contradicting the experimental results.

---

## Learning-Rate Scheduling

The final training process uses:

`ReduceLROnPlateau`

Example configuration:

The final training stage uses `ReduceLROnPlateau` to reduce the learning rate when validation loss stops improving.

If validation loss stops improving, the learning rate is reduced.

This allows larger optimization steps earlier in training and smaller refinement steps later.

---

## Early Stopping

Early stopping monitors validation loss.

If validation performance stops improving for several epochs, training stops automatically.

This helps:

- Reduce unnecessary computation
- Limit overfitting
- Preserve the strongest validation model

The best model is selected using validation loss only.

---

## Model Checkpointing

The notebook saves a complete training checkpoint:

`checkpoints/best_houseprice_mlp.pth`

The checkpoint contains the model state, optimizer state, scheduler state, epoch, validation loss, model configuration, and feature names.

`Model state`, `Optimizer state`, `Scheduler state`, `Epoch`, `Validation loss`, `Model configuration`, `Feature names`

This makes it possible to reconstruct and reload the trained model.

The notebook verifies the checkpoint using the validation set before evaluating the test set.

---

## Preprocessing Artifacts

The preprocessing pipeline is also saved:

`artifacts/preprocessing.joblib`

It stores the encoder, input scaler, target scaler, feature names, numerical feature names, and categorical feature names.

The categorical encoder uses `OneHotEncoder` with unseen-category handling and is fitted only on the training split.

Saving these objects is important because a trained model must receive new data in exactly the same format used during training.

---

## Final Evaluation

The final PyTorch model is evaluated on the held-out test set only after:

1. Data preprocessing
2. Experimental comparison
3. Model configuration selection
4. Final training
5. Early stopping
6. Checkpoint verification

The evaluation metrics are:

### R² Score

R² measures how much of the variation in house prices is explained by the model.

Higher is better.

### Mean Absolute Error

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

MAE represents the average absolute prediction error in USD.

Lower is better.

### Root Mean Squared Error

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

RMSE penalizes large prediction errors more strongly than MAE.

Lower is better.

The notebook prints the final metrics automatically and saves a machine-readable summary to:

`artifacts/final_results.json`

---

## Visualizations

The notebook generates analysis and evaluation figures in:

`analysis_graphs/`

Generated figures include:

`eda_price_distribution.png`, `eda_view_vs_price.png`, `experiment1_weight_initialization.png`, `experiment2_activation_functions.png`, `experiment3_optimizers.png`, `experiment4_regularization.png`, `final_model_learning_curves.png`, `prediction_vs_actual_scatter.png`, `prediction_residual_distribution.png`, `evaluation_classical_vs_deep_learning.png`

The price and prediction plots use dynamic or logarithmic ranges so expensive properties and large errors are not silently hidden.

---

## Held-Out Prediction Examples

The notebook also demonstrates predictions using examples selected from the held-out test split.

The examples include:

- A low-view property
- A high-view property

For each property, the notebook reports:

`City / Zipcode`, `View Rating`, `Waterfront Indicator`, `Actual Price`, `Predicted Price`, `Absolute Error`, `Percentage Error`

These examples are never selected from the training set.

---

## Project Structure

```text
.
├── README.md
├── houseprice.csv
├── Phase3Task2.ipynb
│
├── checkpoints/
│   └── best_houseprice_mlp.pth
│
├── artifacts/
│   ├── preprocessing.joblib
│   └── final_results.json
│
└── analysis_graphs/
    ├── eda_price_distribution.png
    ├── eda_view_vs_price.png
    ├── experiment1_weight_initialization.png
    ├── experiment2_activation_functions.png
    ├── experiment3_optimizers.png
    ├── experiment4_regularization.png
    ├── final_model_learning_curves.png
    ├── prediction_vs_actual_scatter.png
    ├── prediction_residual_distribution.png
    └── ```

---

## Requirements

The project uses Python with the following libraries:

`PyTorch`, `pandas`, `NumPy`, `Matplotlib`, `scikit-learn`, `joblib`, `Jupyter`

Install the required packages with:

```bash
pip install torch pandas numpy matplotlib scikit-learn joblib jupyter
```

---

## Running the Project

Clone the repository and place `houseprice.csv` in the project root.

Start Jupyter:

```bash
jupyter notebook
```

Open:

`Phase3Task2.ipynb`

Then run all cells from top to bottom.

The notebook will:

1. Load and clean the dataset
2. Create engineered features
3. Create grouped train/validation/test splits
4. Encode categorical variables
5. Scale the data
6. Build the PyTorch MLP
7. Compare initialization methods
8. Compare activation functions
9. Compare optimizers
10. Compare regularization methods
11. Select the final model configuration
12. Train with scheduling and early stopping
13. Save and reload the best checkpoint
14. Evaluate the held-out test set
15. Generate analysis and evaluation figures
16. Save preprocessing and result artifacts

---

## Deep Learning Workflow

The complete learning process can be summarized as:


At the core of training:

```python
optimizer.zero_grad()
predictions = model(x)
loss = criterion(predictions, y)
loss.backward()
optimizer.step()
```

The model repeatedly predicts, measures its error, calculates gradients, updates its weights, and improves over time.

---

## Key Takeaways

- MLPs can be applied to structured tabular regression tasks.
- Data leakage can produce unrealistic results and must be prevented.
- Preprocessing objects must be fitted only on training data.
- Scaling helps neural-network optimization.
- Weight initialization affects convergence.
- Activation functions determine the non-linearity of the network.
- Optimizers control how gradients update model parameters.
- Dropout and BatchNorm can improve generalization and training behavior.
- Validation data should guide model selection.
- The test set should remain separate until final evaluation.
- Early stopping and learning-rate scheduling improve training efficiency.
- Checkpointing preserves the best model rather than only the final epoch.
- Experimental results are specific to the dataset and configuration and should not be treated as universal rules.

---

## License

This repository is intended for educational and research purposes as part of Phase 3 Task 2 at JRCAI.

The dataset remains subject to its original source and usage terms.
