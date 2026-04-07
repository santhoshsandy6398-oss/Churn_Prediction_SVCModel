import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
mean_squared_error, r2_score)
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop(columns=['customerID', 'gender'], inplace=True)


df.head()
df.tail()
df.info()
df.describe()
df.drop_duplicates(inplace=True)
threshold = len(df)*0.5
df.dropna(thresh=threshold, axis=1, inplace=True)
df.shape
df.dtypes
#df.corr()['churn'] 

df.head()
for col in df.columns:
    print(f'{col}: {df[col].unique()}')


df['Partner']=df['Partner'].map({'Yes':1, 'No':0})
df['Dependents']=df['Dependents'].map({'Yes':1, 'No':0})
df['PhoneService']=df['PhoneService'].map({'Yes':1, 'No':0})
df['MultipleLines']=df['MultipleLines'].map({'Yes':1, 'No':0, 'No phone service':2})
df['InternetService']=df['InternetService'].map({'DSL':1, 'Fiber optic':2, 'No':0})
df['OnlineSecurity']=df['OnlineSecurity'].map({'Yes':1, 'No':0, 'No internet service':2})
df['OnlineBackup']=df['OnlineBackup'].map({'Yes':1, 'No':0, 'No internet service':2})
df['DeviceProtection']=df['DeviceProtection'].map({'Yes':1, 'No':0, 'No internet service':2})
df['TechSupport']=df['TechSupport'].map({'Yes':1, 'No':0, 'No internet service':2})
df['StreamingTV']=df['StreamingTV'].map({'Yes':1, 'No':0, 'No internet service':2})
df['StreamingMovies']=df['StreamingMovies'].map({'Yes':1, 'No':0, 'No internet service':2})
df['Contract']=df['Contract'].map({'Month-to-month':1, 'One year':2, 'Two year':3})
df['PaperlessBilling']=df['PaperlessBilling'].map({'Yes':1, 'No':0})
df['PaymentMethod']=df['PaymentMethod'].map({'Electronic check':1, 'Mailed check':2, 'Bank transfer (automatic)':3, 'Credit card (automatic)':4})
df['Churn']=df['Churn'].map({'Yes':1, 'No':0})


df.head()



df.dtypes
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
df.dtypes
df.corr()['Churn'].sort_index(ascending=False)



df.drop(columns=['MultipleLines', 'PhoneService'], inplace=True)



df.corr()['Churn']


X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Churn']), df['Churn'], test_size=0.2, random_state=42)




scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


import numpy as np

# Identify rows in X_train that contain NaN values
nan_rows_mask_train = np.any(np.isnan(X_train), axis=1)

# Filter X_train and y_train to remove rows with NaN values
X_train = X_train[~nan_rows_mask_train]
y_train = y_train[~nan_rows_mask_train]

# Identify rows in X_test that contain NaN values
nan_rows_mask_test = np.any(np.isnan(X_test), axis=1)

# Filter X_test and y_test to remove rows with NaN values
X_test = X_test[~nan_rows_mask_test]
y_test = y_test[~nan_rows_mask_test]

print(f"X_train shape after dropping NaNs: {X_train.shape}")
print(f"y_train shape after dropping NaNs: {y_train.shape}")
print(f"X_test shape after dropping NaNs: {X_test.shape}")
print(f"y_test shape after dropping NaNs: {y_test.shape}")


models = {'Decision Tree Classifier' : DecisionTreeClassifier(max_depth=3, random_state=42),
           'Random Forest Classifier' : RandomForestClassifier(n_estimators=100,max_depth=5,random_state = 42),
        'KNN Classifier' : KNeighborsClassifier(),
        'SVM' : SVC(kernel='linear')} # Changed SVC to SVR for regression

results={}



for name, model in models.items():
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    recall_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

    # Store results
    results[name] = {
        'Precision (Train)': precision_train,
        'Recall (Train)': recall_train,
        'F1-score (Train)': f1_train,
        'Precision (Test)': precision_test,
        'Recall (Test)': recall_test,
        'F1-score (Test)': f1_test
    }

print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")



from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize and train the best model (SVM Classifier)
best_model = SVC(kernel='linear', random_state=42)
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train_best = best_model.predict(X_train)
y_pred_test_best = best_model.predict(X_test)

# Calculate and print metrics for the best model
print("SVM Classifier Performance:")

precision_train_best = precision_score(y_train, y_pred_train_best, average='weighted', zero_division=0)
recall_train_best = recall_score(y_train, y_pred_train_best, average='weighted', zero_division=0)
f1_train_best = f1_score(y_train, y_pred_train_best, average='weighted', zero_division=0)

precision_test_best = precision_score(y_test, y_pred_test_best, average='weighted', zero_division=0)
recall_test_best = recall_score(y_test, y_pred_test_best, average='weighted', zero_division=0)
f1_test_best = f1_score(y_test, y_pred_test_best, average='weighted', zero_division=0)

print(f"  Precision (Train): {precision_train_best:.4f}")
print(f"  Recall (Train): {recall_train_best:.4f}")
print(f"  F1-score (Train): {f1_train_best:.4f}")
print(f"  Precision (Test): {precision_test_best:.4f}")
print(f"  Recall (Test): {recall_test_best:.4f}")
print(f"  F1-score (Test): {f1_test_best:.4f}")

# Display Confusion Matrix for the Test Set
cm = confusion_matrix(y_test, y_pred_test_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM Classifier (Test Set)')
plt.show()



import pandas as pd

# Create a sample new customer (ensure columns match X_train in order and type)
# Refer to df.columns before dropping 'Churn' to understand the order
# Original columns after initial drop: 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'

# NOTE: 'MultipleLines' and 'PhoneService' were dropped, so exclude them here.

# Example values for a hypothetical customer
# This is a sample, you can change these values.
new_customer_data = {
    'SeniorCitizen': [0], # Not a senior citizen
    'Partner': [1],       # Has a partner
    'Dependents': [0],    # No dependents
    'tenure': [24],       # 24 months tenure
    'InternetService': [2], # Fiber optic (2)
    'OnlineSecurity': [1],  # Yes (1)
    'OnlineBackup': [1],    # Yes (1)
    'DeviceProtection': [0],# No (0)
    'TechSupport': [1],     # Yes (1)
    'StreamingTV': [1],     # Yes (1)
    'StreamingMovies': [1], # Yes (1)
    'Contract': [1],        # Month-to-month (1)
    'PaperlessBilling': [1],# Yes (1)
    'PaymentMethod': [1],   # Electronic check (1)
    'MonthlyCharges': [85.0],
    'TotalCharges': [2000.0]
}

new_customer_df = pd.DataFrame(new_customer_data)

print("New customer data:")
display(new_customer_df)

# Scale the new customer data using the *trained* scaler
new_customer_scaled = scaler.transform(new_customer_df)

# Make prediction using the best model
prediction = best_model.predict(new_customer_scaled)

# Interpret the prediction
churn_status = 'Yes' if prediction[0] == 1 else 'No'
print(f"\nPredicted Churn for the new customer: {churn_status} (0 = No Churn, 1 = Churn)")




