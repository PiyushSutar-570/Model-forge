# Model Forge - Customer Churn Prediction

A machine learning pipeline built to predict customer churn using classification models. It benchmarks Logistic Regression, Decision Tree, and Random Forest classifiers to identify key churn indicators.

## Features

- Data preprocessing: Standard scaling for numerical features & One-Hot Encoding for categorical attributes.
- Benchmarking across 3 algorithms (Logistic Regression, Decision Tree, Random Forest).
- Automated plot generation (Confusion Matrix, ROC Curve, Feature Importance).
- Summary report export (`.docx` / text format).

## Repository Structure

```
model-forge/
├── main.py                  # End-to-end pipeline execution script
├── ml_model_notebook.ipynb  # Interactive analysis notebook
├── customer_churn_data.csv  # Customer churn dataset
├── requirements.txt        # Python package dependencies
└── output/                  # Generated plots and evaluation reports
```

## Setup & Execution

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**
   ```bash
   python main.py
   ```

3. **Run Notebook**
   ```bash
   jupyter notebook ml_model_notebook.ipynb
   ```

