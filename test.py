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
