# Census Income Classification: Predicting High Earners (> $50,000)

## 1. Project Overview and Objective

This project addresses the binary classification task of predicting whether an individual’s total person income exceeds **$50,000 per annum** using U.S. Census Bureau data.

The objective of this submission is to provide a complete, documented, and professionally evaluated solution that meets high standards of **code accessibility** and **rigorous validation**, directly addressing the requirements for enterprise-level data science work.

### Key Focus Areas

1.  **Functional Pipeline:** Structuring the end-to-end notebook using clear Python functions to ensure code logic is clean, reusable, and easy to evaluate.
2.  **Imbalance Correction:** Explicitly mitigating the challenge where only **6.20%** of the population belongs to the positive class (Income $>\$50K$).
3.  **Rigorous Validation:** Adhering to the mandated `learn`/`test` split protocol for unbiased final evaluation.

---

## 2. Data Context and Constraints

The analysis utilises data derived from the U.S. Census Bureau database, featuring 40 attributes.

### Dataset Split and Storage

Due to repository constraints, all files reside in the root directory. The data files required for execution are contained within the compressed file, **`data_files.zip`**, which includes:

* **`census_income_learn.csv`**: Used for all **training, internal validation, and Exploratory Data Analysis (EDA)**.
* **`census_income_test.csv`**: Used strictly as the **final, unseen hold-out set** for scoring the final model performance.

The full data dictionary and attribute constraints are provided in **`census_income_metadata.txt`** in the root directory.

### Critical Technical Challenge: Class Imbalance

The severe class imbalance is the central technical challenge. The data exhibits a strong skew, with only **6.20%** of instances belonging to the $>\$50K$ class. This demands specific mitigation techniques to produce a valuable, unbiased model.

### Data Governance

The documentation is clear that the **`instance weight`** attribute, while vital for population-level analysis, is **excluded** from the classifier training process. This project adheres to this separation of use.

---

## 3. Technical Approach and Workflow

### 3.1. Two-Notebook Workflow for Clarity

The analysis is segmented into two clear, readable stages, demonstrating a clear separation of concerns:

* **`notebooks/1.0-EDA-Report.ipynb`**: Dedicated to initial data understanding, visualisations, and exploratory analysis of the `learn` dataset.
* **`notebooks/2.0-Production-Pipeline.ipynb`**: This notebook contains the entire functional pipeline. All major steps (data cleaning, feature engineering, training, and final evaluation) are **encapsulated in distinct Python functions** to ensure the code logic is clear, reusable, and easy to evaluate.

### 3.2. Class Imbalance Mitigation (XGBoost)

The final model employs an **XGBoost Classifier**, selected for its performance and robustness. The class imbalance is handled using a reliable, explicit sampling technique:

* **Strategy:** A **Resampling Technique** (specifically **Oversampling** of the minority class using `sklearn.utils.resample`) is applied to the training data. This process duplicates existing instances of the high-earning class until the training set is more balanced, ensuring the model effectively learns the characteristics of the minority group.

### 3.3. Evaluation Metrics: F1-Score Priority

The model's success is judged on its utility for identifying the high-value segment of high earners, which necessitates going beyond simple accuracy or ROC-AUC:

| Metric | Business Focus |
| :--- | :--- |
| **Precision** | **Trustworthiness:** Of all records predicted as >$50K, how many were correctly identified? |
| **Recall** | **Completeness:** Of all individuals who *actually* earned >$50K, how many did the model successfully find? |
| **F1-Score** | **Primary Objective:** The balanced measure of Precision and Recall for the minority class, indicating a truly robust and deployable model. |

---

## 4. Setup and Execution

### Dependencies

All necessary packages are listed in `requirements.txt`:

```bash
# Core Data Manipulation
pandas>=1.3.0
numpy>=1.21.0

# Visualisation and EDA
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning and Preprocessing (includes resample functionality)
scikit-learn>=1.0.0
xgboost>=1.5.0

# Notebook Execution and Environment
ipykernel>=6.0.0
jupyterlab>=3.0.0
```
### Installation
Clone the repository:

```bash
git clone [https://github.com/YourUsername/census-income-classification.git](https://github.com/YourUsername/census-income-classification.git)
cd census-income-classification
```

###Data Preparation

The following step must be executed to prepare the data files (census_income_learn.csv and census_income_test.csv) for use by the notebooks:
Install all required Python dependencies:

```bash
unzip data_files.zip
```
```bash
pip install -r requirements.txt
```
Execution
The entire production process can be executed by running the cells in the notebooks/2.0-Production-Pipeline.ipynb notebook.
