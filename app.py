import subprocess

def run_script(script_name):
    subprocess.run(['python', script_name])

if __name__ == "__main__":
    # Run model_training_saving.py
    print("Running model_training_saving.py...")
    run_script('model_training_saving.py')
    print("model_training_saving.py completed.")

    # Run data_pipeline_bank_churn.py
    print("Running data_pipeline_bank_churn.py...")
    run_script('data_pipeline_bank_churn.py')
    print("data_pipeline_bank_churn.py completed.")
