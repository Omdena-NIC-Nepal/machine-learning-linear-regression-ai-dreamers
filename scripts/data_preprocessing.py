from scipy import stats
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mstats 
import argparse
from sklearn.preprocessing import MinMaxScaler

def preprocess_boston_housing(file_path):
    """
    Preprocesses the Boston Housing dataset and saves the processed dataset in the same directory.
    
    Parameters:
        file_path (str): Path to the file boston_housing.csv.
    """
    # extract directory name from the file path
    directory_name = os.path.dirname(file_path)

    # Load the dataset
    df = pd.read_csv(file_path)

    ## For high skewness data(Logarithmic transformation)
    df_skew = df.copy()

    # For heavily skewed data(note due to the presence of zero in chas and zn log stat wont work on them)
    df_skew['log_lstat']=np.log(df_skew.lstat)
    df_skew['log_crime']=np.log(df_skew.crim)
    df_skew['log_dis']=np.log(df_skew.dis)
    df_skew['log_rad']=np.log(df_skew.rad)
    df_skew['log_medv'] = np.log(df_skew.lstat)

    # for data with zeros and ones wwe will use logit transformation and boxcox transformation
    df_skew['box_zn'], best_lambda = stats.boxcox(df_skew['zn'] + 1)

    # for heavily negatively skewed data
    df_skew['log_b'] = np.log(-df_skew['b'] + df_skew['b'].max() + 1)

    # for moderately skewed data
    df_skew['win_rm'] = mstats.winsorize(df_skew['rm'], limits=[0.05, 0.05])

    # dataframe of heavily skewed data
    df_heavyskew = df_skew.drop(columns=['crim','zn','dis','rad','b','log_lstat','win_rm','medv'])
    df_heavyskew.to_csv(os.path.join(
        directory_name, "data_heavyskew.csv"), index=False)
    
    # dataframe of highly correlated and skewed data
    df_corskew = df_skew.drop(columns=['log_crime', 'box_zn', 'log_dis', 'log_rad', 'log_b', 'lstat', 'rm', 'medv'])
    df_corskew.to_csv(os.path.join(
        directory_name, "data_corskew.csv"), index=False)

    # normalize different datasets and save them as csv
    scaler = MinMaxScaler()
    datasets = {"df": df, "df_heavyskew": df_heavyskew,
                "df_corskew": df_corskew}
    for name, dataset in datasets.items():
        normalized = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
        normalized.to_csv(os.path.join(directory_name, f"{name}_normalized.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the Boston Housing dataset and save the processed file.")
    parser.add_argument("file_path", type=str,
                        help="Path to the Boston Housing dataset file (CSV).")

    args = parser.parse_args()

    preprocess_boston_housing(args.file_path)
