# Air Quality Prediction End-to-End Project

## Overview

This is an end-to-end project demonstrating prediction of the Air Quality Index (AQI) based on user input using machine learning. The project is implemented in Python and HTML, with Flask used to convert it into a web application.

## Components

### 1. Data Ingestion
Data ingestion takes a .csv dataset, splits it into training and testing sets, saves it into the artifacts folder with the full unsplit dataset, and returns the training and testing arrays.

### 2. Data Transformation
Data transformation removes outliers from the data, normalizes, and scales the data using Pipeline, FunctionTransformer, and ColumnTransformer.

### 3. Model Trainer
Model Trainer trains the data on all the below-mentioned machine learning models with the best parameters obtained using hyperparameter training. At last, it saves the pickle file of the best model to the artifacts folder for later use.

## Pipeline

### Predict Pipeline
The predict pipeline takes the user input, converts the data into a Pandas data frame, and uses this data frame to predict the Air Quality Index. Based on the predicted Air Quality Index,categorizes it into different categories.

## Machine Learning Algorithms Used

The project employs the following machine learning algorithms for predicting the Air Quality Index:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- AdaBoost

## Core Libraries Used

- pandas
- numpy
- scikit-learn
- Flask

## Technologies Used
- Python
- HTML
- Flask

## Setup and Installation

### Prerequisites
- Python

## Usage

- Prediction of the Air quality index with its Air Quality category.

## Web App Snippet

![image](https://github.com/sumit936/End-to-End-Air-Pollution-Prediction/assets/47924474/76bd5037-93ca-42c4-8340-bbaee114b776)

---
let me know if you have any specific additions or changes in mind.
