# credit-risk-classification

# Overview of the Analysis

The primary objective of this analysis was to develop and evaluate machine learning models capable of predicting the credit risk of loan applicants. Specifically, the goal was to classify loans into two categories: healthy (low risk) and high-risk. Accurate classification of loans is crucial for financial institutions to minimize the risk of default and make informed lending decisions.

## Financial Information and Prediction Target

The dataset utilized for this analysis comprised various financial information about loan applicants. This included variables such as "loan size", "interest rate", "borrower income", "debt-to-income", "num-of-accounts", "derogatory marks", and "total debt". The target variable to predict was the loan_status, which was binary: 0 for healthy loans and 1 for high-risk loans. The aim was to create a model that could accurately predict this binary outcome based on the input features.

## Basic Information About the Variables

To understand the distribution of the target variable, a value_counts analysis was performed. The results indicated the following distribution:

- 75036 Healthy loans (0): X instances
- 2500 High-risk loans (1): y instances

This distribution highlighted any class imbalance in the dataset, which is a crucial consideration for model performance and evaluation.

## Stages of the Machine Learning Process

### Data Preprocessing:

- The first step was identifying that the target variable used for predictions was a categorical variable and assigning that as the y (loan status). The other features were used as X to train the model to find patterns to predict y.
- No scaling was done to the dataset, although it wouldn't result in worse results.
- It was also important to check for an imbalance between healthy loans (0) and high-risk loans (1) in the loan status column, which the model is being trained to predict.

### Instantiate Model:

- The logistic regression model was instantiated to be trained.

### Fit or Train Model:

- The dataset was split into training and testing sets using X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y), ensuring a 75% training and 25% testing split. 
- stratify=y was used to account for the imbalance in the healthy and high-risk loans in the loan status column, ensuring an even distribution of classes in the training and testing sets. 
- The model was then fit/trained using model.fit(X_train, y_train) on the training set.

### Predict Model:

- Predictions were made on the test set using test_predictions = model.predict(X_test) to evaluate how the model would perform with new data (data the model hasn't seen yet) and its ability to predict/classify healthy vs. high-risk loans based on the features it was trained on.

### Evaluate Model:

- Evaluation metrics like accuracy, precision, recall, and F1-score were used to compare model performance.

### Methods Used

The primary method used in this analysis was LogisticRegression due to its effectiveness in binary classification problems. Additionally, other algorithms like Decision Trees and Random Forests were explored to compare their performance. Logistic Regression was particularly chosen for its simplicity, interpretability, and good performance with well-separated classes.

In conclusion, this comprehensive analysis involved multiple stages of the machine learning process, from data preprocessing to model evaluation. The use of LogisticRegression and other algorithms provided valuable insights into predicting credit risk, forming a robust foundation for developing reliable credit risk assessment models.

## Results

The logistic regression model demonstrated excellent performance with the following metrics:

- Accuracy: 0.99: The model achieved a very high overall accuracy, correctly predicting the loan status 99% of the time.

- Healthy Loans (0):
    Precision: 0.99: The model perfectly identified all loans classified as healthy. There were almost no false positives.
    Recall: 1.0: The model correctly identified all actual healthy loans. There were no false negatives.
    F1 Score: 1.0: The harmonic mean of precision and recall is perfect, indicating flawless identification of healthy loans.

- High-Risk Loans (1):
    Precision: 0.91: Out of all the loans the model predicted as high-risk, 91% were actually high-risk. This means the model was very accurate in identifying high-risk loans.
    Recall: 0.85: The model correctly identified 85% of actual high-risk loans, but 15% of high-risk loans were incorrectly classified as healthy.
    F1 Score: 0.88: The F1 score, which balances precision and recall, indicates that the model effectively predicts high-risk loans while minimizing false positives and false negatives.

## Summary

The logistic regression model performed exceptionally well in predicting both healthy and high-risk loans. Its high accuracy and perfect precision, recall, and F1 score for healthy loans make it an excellent choice for scenarios where correctly identifying healthy loans is crucial.

However, for high-risk loans, the model still maintains strong performance but with slightly lower precision and recall. Given the financial implications of misclassifying high-risk loans, it is essential to consider the context of the problem:

- If the priority is to identify healthy loans accurately, this logistic regression model is highly recommended due to its perfect scores for healthy loan classification.
- If identifying high-risk loans is more critical, the model still performs well but could benefit from further optimization or combining with additional techniques to reduce the misclassification rate of high-risk loans.

In conclusion, the logistic regression model is recommended for its overall high performance. However, if minimizing the misclassification of high-risk loans is paramount, additional measures or further tuning may be required to enhance its effectiveness in that area.
