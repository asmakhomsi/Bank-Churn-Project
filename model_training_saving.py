import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler #for the scalling
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import os
from joblib import dump

# reading the data
file_path = r"input\bank_churn_90_training.csv"
df = pd.read_csv(file_path, index_col=0)

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
cols_to_scale = [col for col in cols_to_scale if col not in new_cols_ohe and col != "Exited" and col not in like_num]
scaler = RobustScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

#Training
# Separate features (X) and target variable (y)
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)

# Initialize LightGBM classifier
lgbm_model = LGBMClassifier(random_state=123456)

# Perform KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=123456)
cv_results = cross_val_score(lgbm_model, X_train, y_train, cv=kfold)

# Fit LightGBM model and make predictions
lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Directory path where you want to save the model
directory = r"output"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Assuming your model is named "lgbm_model"
model_path = os.path.join(directory, "bank_churn_final_model.joblib")
dump(lgbm_model, model_path)



