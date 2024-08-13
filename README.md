 Predictive Maintenance Analysis using Machine Learning

This project involves the analysis of predictive maintenance data using machine learning techniques. The dataset used was sourced from **Kaggle**, a popular platform for data science projects.

## Project Overview

The objective is to develop models that predict failures and optimize maintenance schedules. The key steps include:

1. **Data Import and Cleaning:**
   - Imported the `predictive_maintenance.csv` dataset and cleaned it by removing unnecessary columns, handling missing values, and eliminating duplicates.

2. **Exploratory Data Analysis (EDA):**
   - Conducted EDA to understand feature distributions and relationships.
   - Used histograms, count plots, pair plots, and correlation matrices for visualization.

3. **Data Preprocessing:**
   - Applied one-hot encoding for categorical features and scaling for numerical features.
   - Split the data into training and testing sets.

4. **Machine Learning Models:**
   - Developed and evaluated models, including Logistic Regression and SVC.
   - Used RandomUnderSampler and SMOTE to address class imbalance.

5. **Model Evaluation:**
   - Assessed model performance using accuracy scores.
   - Compared results before and after resampling to improve robustness.

6. **Model Deployment:**
   - Saved the trained logistic regression model for future use.

## Visualizations

Included visualizations:
- Feature distribution histograms.
- Pair plots for feature relationships.
- Correlation matrices.
- Regression plots.

## Tools and Libraries

- **Python**: Primary language.
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn.
- **Data Source**: Kaggle

## How to Use

1. Clone the repository.
2. Install the required libraries.
3. Run `codMainteance.py` to reproduce the analysis.
4. Use the saved model for predictions.
