# 🚢 Titanic Survival Prediction - Machine Learning Project

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning techniques.

The model is built using the Titanic dataset and follows an industrial-level ML workflow including:

- Data Cleaning
- Feature Engineering
- Model Comparison
- Cross Validation
- Pipeline Integration
- Model Deployment Ready Structure

---

## 🎯 Problem Statement

Build a binary classification model to predict passenger survival (`Survived`) using demographic and travel information.

Target Variable:
- `Survived` (0 = No, 1 = Yes)

---

## 📊 Dataset Information

The dataset contains 891 rows and the following important features:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## 🧠 Feature Engineering

The following new features were created:

- **FamilySize** = SibSp + Parch + 1
- **Title** extracted from passenger names (Mr, Mrs, Miss, Rare, etc.)

High-cardinality columns like:
- Name
- Ticket
- PassengerId
- Cabin

were removed to avoid overfitting.

---

## 🛠️ Data Preprocessing

- Missing `Age` filled with median
- Missing `Embarked` filled with mode
- Categorical variables encoded using One-Hot Encoding
- Train-test split with stratification

---

## 🤖 Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest (Selected Model)

---

## 📈 Model Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Cross Validation (5-fold)

Final Model Accuracy: **~79%**

---

## 🚀 Model Deployment

The trained model and feature columns were saved using:

```python
joblib.dump(model, "models/titanic_model.pkl")
joblib.dump(X.columns, "models/feature_columns.pkl")
