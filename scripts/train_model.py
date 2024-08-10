import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import argparse


def train_linear_regression_model(data_folder):
    """
    Trains a linear regression model on the datasets from the given path of data folder and saves it in the specified directory.
    
    Parameters:
        data_folder (str): Path to the folder containing CSV files of the datasets.
    """

    # Define the path for the models folder
    models_folder = os.path.join(data_folder, "models")

    # Ensure the "models" directory exists
    os.makedirs(models_folder, exist_ok=True)

    # Iterate over each file in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            # Construct file paths
            data_path = os.path.join(data_folder, filename)
            model_name = os.path.splitext(filename)[0] + '_model.pkl'
            model_path = os.path.join(models_folder, model_name)

            # Load the dataset
            data = pd.read_csv(data_path)

            # Define features (X) and target (y)
            target = "log_medv" if "log_medv" in data.columns else "medv"
            X = data.drop(columns=[target])
            y = data[target]

            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Initialize the Linear Regression model
            model = LinearRegression()

            # Train the model on the training data
            model.fit(X_train, y_train)

            # Save the trained model
            joblib.dump(model, model_path)
            print(f"Model trained and saved as '{model_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a linear regression model on datasets from the given folder and save them.")
    parser.add_argument("data_folder", type=str,
                        help="Path to the folder containing CSV files of the datasets.")

    args = parser.parse_args()

    train_linear_regression_model(args.data_folder)
