# Hello there! Welcome to our Movie Review Analysis: Sentiment, Score, and Critic Classification project!

#Dataset Address:https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

## Overview
This project focuses on analyzing movie reviews using machine learning. It includes three tasks:
1. **Sentiment Classification** — Determine whether a review is positive or negative.
2. **Score Regression** — Predict a numerical score (0–10) associated with a review.
3. **Critic Identification** — Identify whether a review was written by a top critic or a regular user.

We explore two approaches: 
- A **TF-IDF + Traditional ML** pipeline for benchmarking.
- A **BERT + Tree-Based Model** pipeline for contextual, high-performance prediction.

## Files
- `BERT_Movie_Review.ipynb`  
  → Uses `bert-base-uncased` to extract embeddings and feeds them into XGBoost and CatBoost models.

- `TF-IDE_Movie_Review.ipynb`  
  → Builds a baseline pipeline using TF-IDF features with models like Random Forest and Logistic Regression.

- `1513 Final Project Movie Review.pdf`  
  → Final project report detailing the methodology, experiments, results, and analysis.

## Key Results
- **BERT-based models** outperform TF-IDF baselines in sentiment and score prediction, offering better generalization and reduced overfitting.
- **Score regression** using BERT + XGBoost achieved strong alignment with ground truth scores.
- **Top critic classification** remained challenging due to class imbalance, though BERT improved minority class performance.

## Team Members & Roles
- **Kaiwen Zhu** — Team Lead & Integration Engineer  
- **Zhengan Du** — Data Engineer  
- **Shiang He** — Model Engineer  
- **Jiachang Zhang** — Research & Analysis Specialist

## Requirements
- Python 3.x
- Jupyter Notebook
- `transformers`, `scikit-learn`, `xgboost`, `catboost`, `pandas`, `matplotlib`, `seaborn`, `torch`

## How to Run
1. Open either `.ipynb` file in Jupyter Notebook.
2. Ensure all required packages are installed.
3. Execute the cells in sequence to reproduce the results.


