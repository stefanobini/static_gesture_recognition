"""
python3 datasets/utils/split_annotations.py
"""

import os
import pandas as pd
from math import floor
from typing import Tuple


DATASET_PATH = os.path.join("datasets", "MIVIA_HGR")
ANNOTATION_FILENAME = "demo7_rgb.csv"
IN_ANNOTATION_FILE = os.path.join(DATASET_PATH, ANNOTATION_FILENAME)
SETS = {
    "train":0.8,
    "val":0.1,
    "test":0.1
}


def split_dataframe(df:pd.DataFrame, train_rate:float, valid_rate:float, test_rate:float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splid a dataframe in training, validation and test set.

    Parameters
    ----------
    df: pandas.DataFrame
        Pandas dataframe containing the data to split
    train_rate: float
        Relative percentage of the sample to include in training set
    valid_rate: float
        Relative percentage of the sample to include in validation set
    test_rate: float
        Relative percentage of the sample to include in test set

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe containing the training data
    pandas.DataFrame
        Pandas dataframe containing the validation data
    pandas.DataFrame
        Pandas dataframe containing the test data
    """
    n_sample = len(df)

    num_valid = floor(n_sample*valid_rate) if (floor(n_sample*valid_rate)>0 or n_sample<3) else 1
    num_test = floor(n_sample*test_rate) if (floor(n_sample*test_rate)>0 or n_sample<3) else 1
    num_train = n_sample - num_valid - num_test

    train_df = df.iloc[:num_train]
    valid_df = df.iloc[num_train:num_train+num_valid]
    test_df = df.iloc[num_train+num_valid:]

    return {"train":train_df, "val":valid_df, "test":test_df}


def save_df(path:str, set_dict:dict, columns:list):
    set_dict = set_dict.sample(frac=1)
    df = pd.DataFrame(data=set_dict, columns=columns)
    df.to_csv(path_or_buf=path, index=False, sep=';')


''' Read annotation file '''
df = pd.read_csv(IN_ANNOTATION_FILE, sep=';')
columns = df.columns

''' Shuffle the row of the dataframe '''
df = df.sample(frac=1)

''' Split dataset in training, validation and test set '''
sets_df_dict = split_dataframe(df=df, train_rate=SETS["train"], valid_rate=SETS["val"], test_rate=SETS["test"])

''' Store annotation files '''
for set in SETS.keys():
    save_df(path=IN_ANNOTATION_FILE.replace(".csv", "_{}.csv".format(set)), set_dict=sets_df_dict[set], columns=columns)