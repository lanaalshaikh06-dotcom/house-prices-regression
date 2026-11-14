# House Prices Prediction | Machine Learning

## Project Overview
A Machine Learning regression project designed to predict residential house prices (`SalePrice`) using the House Prices dataset.

The project focuses on building a leakage-safe Machine Learning pipeline and implementing custom feature selection on the original 79 input features.

## Key Features
- Exploratory Data Analysis (EDA).
- Missing value imputation and feature preprocessing.
- Custom `OriginalFeatureSelector` transformer.
- Feature Selection inside an ML Pipeline.
- Leakage-safe 5-Fold Cross Validation.
- Comparison of four regression models.
- Hyperparameter optimization using GridSearchCV.
- Final evaluation on an independent Test Set.

## Models Compared
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Feature Selection
A custom transformer based on Random Forest feature importance is used to select the original input features before preprocessing.

Feature selection is fitted independently within each cross-validation training fold to prevent data leakage.

## Preliminary Results

Before hyperparameter tuning, Gradient Boosting achieved:

| Configuration | CV R² | CV RMSE |
|---|---:|---:|
| All 79 Features | 0.8600 | 28,570.62 |
| Selected Features (95%) | 0.8631 | 28,219.73 |

*Final tuned model and Test Set results will be reported after the final execution.*

## Dataset
House Prices: Advanced Regression Techniques

Source: [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

The dataset is not included in this repository.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## How to Run
1. Download `train.csv` from the Kaggle dataset.
2. Place it in the same working directory as the notebook.
3. Install the required Python libraries.
4. Open and run `house_prices_regression.ipynb`.

## Author
**Lana Al-Shaikh**

Machine Learning & AI Developer
