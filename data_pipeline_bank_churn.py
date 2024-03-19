import pandas as pd
from joblib import load
import os
from sklearn.preprocessing import RobustScaler #for the scalling

# reading the data
file_path = r"input\bank_churn_10_pipeline.csv"
df_init = pd.read_csv(file_path, index_col=0)
# Removing variables that will not affect the dependent variable
df=df_init.drop(["Exited"], axis = 1)

#feature engineering
# we standardize tenure with age
df["NewTenure"] = df["Tenure"]/df["Age"]
df["NewCreditsScore"] = pd.qcut(df['CreditScore'], 6, labels = [1, 2, 3, 4, 5, 6])
df["NewAgeScore"] = pd.qcut(df['Age'], 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])
df["NewBalanceScore"] = pd.qcut(df['Balance'].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])
df["NewEstSalaryScore"] = pd.qcut(df['EstimatedSalary'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#one hot encoding
# Variables to apply one hot encoding
list = ["Gender", "Geography"]
df = pd.get_dummies(df, columns =list, drop_first = True)
# Removing variables that will not affect the dependent variable
df = df.drop(["CustomerId","Surname"], axis = 1)
#scalling
new_cols_ohe = ["Gender_Male", "Geography_Germany", "Geography_Spain"]
like_num = [col for col in df.columns if df[col].dtype != 'object' and len(df[col].value_counts()) <= 10]
cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cols_to_scale = [col for col in cols_to_scale if col not in new_cols_ohe and col not in like_num]
scaler = RobustScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Directory path where the model is saved
directory = r"output"
# Model file path
model_path = os.path.join(directory, "bank_churn_final_model.joblib")
# Load the model
loaded_model = load(model_path)

# Define the directory path
directory = r"output"

# Make predictions
predictions = loaded_model.predict(df)

# Add predictions to the DataFrame
df_init['Predicted_Churn'] = predictions

# Select and rearrange columns
result_df = df_init[['CustomerId', 'Surname','Exited', 'Predicted_Churn']]
result_save = df_init[['CustomerId', 'Surname','Predicted_Churn']]

# Save the resulting table to a CSV file in the specified directory
result_save.to_csv(os.path.join(directory, "predicted_results.csv"), index=True)

# Filter the DataFrame where Exited matches Predicted_Churn
matching_customers = result_df[result_df['Exited'] == result_df['Predicted_Churn']]
# Count the number of matching customers
num_matching_customers = len(matching_customers)
# Calculate the percentage of matching customers
percentage_matching_customers = (num_matching_customers / len(result_df)) * 100
print("Percentage of customers where Exited matches Predicted_Churn:", percentage_matching_customers, "%")