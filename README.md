
# Telco Customer Churn Prediction

## Project Overview

An end-to-end machine learning project to predict
customer churn in the telecommunications industry.

## Objectives

- Explore and understand customer behavior
- Prepare data for machine learning
- Train and evaluate classification models
- Track experiments using MLflow
- Deploy predictions through FastAPI
- Containerize the application using Docker

## Technologies

Python, Pandas, NumPy, Scikit-learn,
Matplotlib, Seaborn, MLflow, FastAPI, Docker

## Project Status

In development


### Experiment Tracking with MLflow

Hyperparameter optimization was conducted using Optuna (TPE),
with 50 trials per model and Stratified 5-fold cross-validation.

MLflow tracked 100 optimization runs, recording hyperparameters,
Average Precision, ROC-AUC, and cross-validation variability.

![MLflow Experiment Tracking](figures/mlflow/experiment_tracking.png)
