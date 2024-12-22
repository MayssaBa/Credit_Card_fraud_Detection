Credit Card Fraud Detection
This project focuses on detecting fraudulent transactions in credit card data using machine learning models. The dataset is processed and analyzed to build and evaluate four models, with Random Forest achieving the best performance.

Dataset
The dataset used in this project is the Fraud Detection Dataset available on Kaggle. The dataset contains two files: fraudTrain.csv and fraudTest.csv, representing training and testing data respectively.
5
The dataset is heavily imbalanced, with a small proportion of transactions labeled as fraudulent. To address this, we applied data balancing techniques to improve the performance of our machine learning models:
-Undersampling: Reduced the majority class to match the size of the minority class.

Models Used

The following machine learning models are trained and evaluated:

Logistic Regression: A linear model for binary classification.
K-Nearest Neighbors (KNN): A distance-based algorithm for classification.
Random Forest: An ensemble learning method using decision trees.
Support Vector Machine (SVM): A kernel-based algorithm for classification.

Model Evaluation Metrics:
Precision
Recall
F1 Score
