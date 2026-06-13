# Titanic Survival Prediction

## Overview

This repository documents my machine learning journey on Kaggle's **Titanic: Machine Learning from Disaster** competition.

The objective is to predict whether a passenger survived the Titanic disaster using passenger information such as age, sex, ticket class, fare, family relationships, and other attributes.

Rather than focusing only on model accuracy, this project emphasizes the complete machine learning workflow:

* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Model development
* Hyperparameter tuning
* Model interpretation

---

## Project Progress

### 1. Logistic Regression (Baseline)

The project began with a Logistic Regression baseline model to establish a performance benchmark.

**Results:**

* Training Accuracy: **80.73%**
* Test Accuracy: **79.21%**

---

### 2. XGBoost with Feature Engineering

After establishing a baseline, I implemented an XGBoost classifier and explored multiple feature engineering strategies.

#### Data Preprocessing

* Handled missing Age values using mean imputation
* Removed Cabin due to excessive missing values
* Removed Ticket as an uninformative identifier
* Processed missing Embarked values

#### Feature Engineering Experiments

##### Family Size Feature

Created a new feature:

```python
Family_Size = SibSp + Parch + 1
```

Analyzed survival rates across family sizes and grouped passengers into:

* Is Alone
* Small Family
* Large Family

##### Passenger Title Extraction

Extracted titles from passenger names:

* Mr
* Mrs
* Miss
* Master
* Rare Titles

Rare titles such as Doctor, Reverend, Major, Countess, Lady, and others were grouped into a single **Rare** category.

##### Cabin Investigation

Explored extracting deck information from the Cabin feature.

Experiments included:

* Individual deck encoding
* Deck grouping
* Cabin availability indicators

After testing, these features did not improve model performance and were removed from the final model.

This was an important lesson in feature engineering: not every intuitive feature improves generalization.

---

## XGBoost Hyperparameter Tuning

To reduce overfitting and improve generalization, I experimented with:

* Maximum tree depth
* Number of estimators
* Learning rate
* Subsampling strategies
* Feature importance analysis

Final model configuration:

```python
XGBClassifier(
    max_depth=3,
    n_estimators=200,
    learning_rate=0.1,
    random_state=2
)
```

### Final Results

* Training Accuracy: **91.28%**
* Test Accuracy: **83.71%**

The tuned XGBoost model significantly outperformed the Logistic Regression baseline.

---

## Feature Importance Insights

The most influential features were:

1. Sex
2. Passenger Title (Mr, Miss, Mrs, Master, Rare)
3. Passenger Class (Pclass)
4. Family Size Categories

This demonstrates the importance of thoughtful feature engineering in tabular machine learning problems.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost

---

## Project Roadmap

* [x] Data Cleaning and Preprocessing
* [x] Logistic Regression Baseline
* [x] Feature Engineering
* [x] XGBoost Implementation
* [x] Hyperparameter Tuning
* [x] Feature Importance Analysis

---

## Dataset

Kaggle Competition:

**Titanic: Machine Learning from Disaster**

---

## Key Learnings

Through this project I learned:

* Data preprocessing techniques
* Handling missing values
* Feature engineering for structured datasets
* One-hot encoding
* Model evaluation and train-test splitting
* Gradient boosting with XGBoost
* Hyperparameter tuning
* Feature importance interpretation
* The importance of experimentation and iterative improvement

---

## Author

**Abeer Malviya**

Machine Learning Student | Exploring Deep Learning, Computer Vision, and AI Research
