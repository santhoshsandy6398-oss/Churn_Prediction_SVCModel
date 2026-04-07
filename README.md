📊 Telco Customer Churn Prediction
📌 Project Overview
Customer churn refers to customers discontinuing a service.
In the telecom industry, churn prediction helps identify customers who are likely to leave so that proactive retention strategies can be applied.
In this project, I used the Telco Customer Churn dataset to build and evaluate multiple machine learning classification models to predict whether a customer is likely to churn.

🎯 Objective

Predict whether a customer will churn or not churn
Compare multiple classification models
Select the best-performing model based on precision, recall, and F1‑score


📂 Dataset

Source: Telco Customer Churn Dataset
Target Variable: Churn (Yes / No)
Features Include:

Customer demographics
Account information
Service usage details
Contract, payment, and billing information




⚙️ Machine Learning Workflow

Data loading and inspection
Data preprocessing and encoding
Train-test split
Model training
Model evaluation using classification metrics


🤖 Models Evaluated
The following classification models were trained and evaluated:

Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM – Linear Kernel)


📈 Evaluation Metrics
To ensure a fair comparison, models were evaluated on both training and test datasets using:

Precision
Recall
F1‑Score


Note: F1‑Score was the primary metric due to the class imbalance present in churn data.


✅ Model Evaluation Results (Test Data)



































ModelPrecisionRecallF1‑ScoreDecision Tree0.76330.78210.7584Random Forest0.77540.79070.7758KNN0.75220.75790.7547✅ SVM0.77700.78710.7803

🏆 Model Selection
The Support Vector Machine (SVM) model was selected as the final model because:

It achieved the highest F1‑score on test data
It demonstrated balanced precision and recall
It showed good generalization with minimal overfitting

Random Forest also performed well and was used as a strong baseline for comparison.

🧠 Key Learnings

F1‑score is more reliable than accuracy for churn prediction
SVM performs well on tabular classification problems
Comparing multiple models helps identify the best trade-off between bias and variance
Feature preprocessing and evaluation strategy significantly impact results


🛠️ Tools & Technologies

Python
Scikit‑learn
Pandas & NumPy
Jupyter Notebook


🚀 Future Improvements

Hyperparameter tuning for SVM and Random Forest
Feature importance analysis
Handling class imbalance using techniques like SMOTE
Model deployment using Flask or FastAPI


📌 Conclusion
This project demonstrates end‑to‑end churn prediction using machine learning, highlighting model comparison, proper evaluation metrics, and informed model selection for real‑world business problems.
