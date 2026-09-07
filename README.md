# E-Commerce Customer Intelligence with PySpark, Databricks & GenAI

An end-to-end e-commerce analytics and machine learning project using **PySpark, Databricks, Spark ML and MLflow**, with a GenAI prompt prototype for converting structured analytics into business insights.

## Project Overview

This project analyzes 25,000 e-commerce customer sessions to understand purchasing behavior, identify patterns in the conversion funnel, and build a machine learning model to predict whether a session results in a purchase.

The project demonstrates a practical data science workflow in Databricks:

**Raw Data → PySpark Analysis → Feature Engineering → Spark ML → Model Evaluation → MLflow → GenAI Business Insight Prototype**

## Objectives

* Analyze e-commerce customer session behavior using PySpark
* Understand the purchase and cart conversion funnel
* Explore customer engagement and session characteristics
* Identify high-performing marketing channels and product categories
* Build a purchase prediction model using Spark ML
* Track and reload the trained model using MLflow
* Demonstrate how structured analytical results can be passed to an LLM through a business-focused prompt

## Technologies

* **Python**
* **PySpark**
* **Databricks Free Edition**
* **Spark ML**
* **MLflow**
* **SQL**
* **Git / GitHub**
* **GenAI / LLM Prompting**

## Dataset

The project uses the **Indian E-Commerce Customer Behavior & Purchase** dataset from Kaggle.

The dataset contains 25,000 e-commerce interaction records and 29 columns covering:

* Customer and session information
* Device type
* User type
* Marketing channel
* Product category
* Price and quantity
* Discounts
* Revenue
* Pages viewed
* Time spent on site
* Cart activity
* Purchase behavior
* Session duration
* Location

The dataset is used for educational and portfolio purposes.

The original dataset is not included in this repository.

## Data Analysis

PySpark was used for data ingestion, transformation, aggregation and exploratory analysis.

### Purchase Funnel

Key findings:

* **25,000** sessions were analyzed.
* Overall purchase rate was **22.46%**.
* **64.47%** of sessions resulted in an item being added to the cart.
* **34.85%** of sessions with an item added to the cart resulted in a purchase.
* Cart abandonment was approximately **42.00%**.

### Session Duration

Session duration showed differences in purchasing behavior:

| Session Duration | Purchase Rate | Average Revenue |
| ---------------- | ------------: | --------------: |
| Very Short       |        20.24% |          345.85 |
| Short            |        22.79% |          418.52 |
| Long             |        23.62% |          435.06 |
| Very Long        |        23.22% |          419.38 |

Long sessions had the highest purchase rate, while very short sessions had the lowest.

These results describe associations in the dataset and should not be interpreted as causal relationships.

### Product Categories

* Product category **6** had the highest purchase rate at **24.68%**.
* Product category **2** generated the highest total revenue.

The category values are encoded in the original dataset, so no business meaning is assigned to the numeric category IDs without supporting documentation.

## Machine Learning

A **Logistic Regression** classifier was developed using PySpark ML.

### Features

The model used:

* Device type
* User type
* Marketing channel
* Product category
* Unit price
* Quantity
* Discount percentage
* Pages viewed
* Time on site
* Added to cart

Potentially leakage-prone variables such as revenue and cart abandonment were excluded from the first model.

### ML Pipeline

The Spark ML pipeline included:

1. Categorical feature indexing
2. One-hot encoding
3. Feature vector assembly
4. Logistic Regression
5. Prediction and evaluation

### Evaluation

The dataset was split into:

* **80% training**
* **20% testing**
* Random seed: **42**

Results:

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 0.7690 |
| AUC         | 0.7632 |
| Weighted F1 | 0.6688 |

Because the target variable is imbalanced, accuracy alone is not sufficient for evaluating model performance. Additional metrics such as AUC, precision, recall and F1 were considered.

## MLflow Experiment Tracking

MLflow was used to track the machine learning experiment.

The following were logged:

* Model type
* Train/test split
* Random seed
* Accuracy
* AUC
* Weighted precision
* Weighted recall
* Weighted F1
* Trained Spark ML model

The logged model was subsequently reloaded from MLflow and used to generate predictions. The reloaded model reproduced the original test accuracy.

## GenAI Component

A small GenAI prototype was developed to demonstrate how structured analytical results can be transformed into a business-oriented LLM prompt.

The session-duration metrics were converted into JSON and inserted dynamically into a prompt containing instructions to:

* Identify key business insights
* Provide a practical recommendation
* State an analytical caveat
* Avoid causal claims
* Use only the supplied data

The Databricks-hosted LLM endpoint was unavailable in the Free Edition workspace used for this project. Therefore, the GenAI component is presented as a **prompt prototype rather than an executed LLM workflow**.

## Project Architecture

```text
                    ┌──────────────────┐
                    │   Kaggle Dataset │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Databricks    │
                    │                  │
                    │     PySpark      │
                    │  Data Ingestion  │
                    │      & EDA       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Engineering      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Spark ML       │
                    │ Logistic         │
                    │ Regression       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Model Evaluation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     MLflow       │
                    │ Experiment       │
                    │ Tracking         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ GenAI Prompt     │
                    │ Business Insight │
                    │ Prototype        │
                    └──────────────────┘
```

## Repository Structure

```text
ecommerce-customer-intelligence-databricks/
│
├── notebooks/
│   └── 01_Data_Ingestion_PySpark.ipynb
│
├── data/
│   └── README.md
│
├── README.md
│
└── .gitignore
```

## How to Reproduce

1. Download the dataset from Kaggle.
2. Upload the CSV to a Databricks Volume.
3. Open the Databricks notebook in this repository.
4. Update the dataset path if necessary.
5. Run the notebook from top to bottom.
6. Review the EDA results and Spark ML evaluation.
7. Open the MLflow experiment to inspect the logged model and metrics.

## Limitations

* The dataset is synthetic and may not represent real-world customer behavior.
* Numeric category and channel codes are not assigned business meanings without additional documentation.
* The analysis identifies associations rather than causal relationships.
* The classification dataset is imbalanced toward non-purchase sessions.
* The GenAI component was implemented as a prompt prototype because the required Databricks-hosted LLM endpoint was unavailable in the Free Edition workspace.

## Future Improvements

Possible extensions include:

* Hyperparameter tuning and cross-validation
* Comparison with tree-based Spark ML models
* Feature importance and model explainability
* More robust handling of class imbalance
* Automated model evaluation
* Deployment of the model through a suitable serving environment
* Execution of the GenAI component using an available LLM endpoint

## Author

**Haritha Retnakaran**

Data Scientist / Data Analyst

This project was developed as a portfolio project to demonstrate practical experience with PySpark, Databricks, Spark ML, MLflow and GenAI concepts.
