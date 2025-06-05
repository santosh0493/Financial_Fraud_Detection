# Financial Fraud Detection

A machine learning project for detecting financial fraud from textual data.

## Overview
This project implements multiple approaches to detect fraudulent financial statements using NLP techniques and machine learning models.

## Models Implemented
- **Logistic Regression** with Bag-of-Words
- **SVM** with TF-IDF
- **Random Forest** with TF-IDF
- **Neural Network** models
- **FinBERT** based models
- **XGBoost**

## Features
- Text preprocessing with NLTK
- Multiple feature extraction methods (BoW, TF-IDF)
- Various ML classifiers for comparative performance
- Evaluation metrics including accuracy and classification reports

## File Structure
- `Bag_of_words_logisitic regression.py`: Implementation of basic NLP approaches
- `ee964-ann.xpynb`: Neural network implementation
- `ee964-eda.xpynb`: Exploratory data analysis
- `ee964-finbert.xpynb` & `ee964-finbert-new.xpynb`: FinBERT model implementations
- `ee964-svm.ipynb`: Support Vector Machine implementation
- `ee964-xgboost.xpynb`: XGBoost model implementation

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- NLTK
- PyTorch (for FinBERT and neural network models)

## Usage
Run individual model files to see their implementation and performance metrics.