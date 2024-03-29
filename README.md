This project aims to predict customer churn for a bank using machine learning techniques.

## Introduction
Customer churn, also known as customer attrition, is a critical metric for businesses, including banks. Identifying customers who are likely to churn allows banks to take proactive measures to retain them, thereby reducing revenue loss.
This project utilizes machine learning techniques to predict customer churn based on various features such as credit score, age, tenure, balance, and more.
The model used for predicting customer churn is based on the LightGBM classifier. The accuracy achieved by the model on the test dataset is 0.8655555555555555.

## Requirements
To run this project, you need to have Python installed on your system.
Make sure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- scikit-learn
- lightgbm
- joblib

You can install these libraries using pip:
pip install pandas numpy matplotlib scikit-learn lightgbm joblib

## Usage
Once you have installed the dependencies, you can use the provided scripts to train the model and make predictions.
run the `app.py`

## Output
After running the `app.py` script, you will find the following output files in the `output` directory:
- `bank_churn_final_model.joblib`: This file contains the trained LightGBM model saved using joblib.
- `predicted_results.csv`: This CSV file contains the predicted churn results generated by the model.
You can locate these files in the `output` directory of the project.
