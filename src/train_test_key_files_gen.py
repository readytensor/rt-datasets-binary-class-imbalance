import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from typing import Tuple, List

import utils
import paths

def create_stratified_kfolds(
    dataset: pd.DataFrame,
    target_col: str,
    num_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits the dataset into k stratified folds.
    
    Args:
        dataset (pd.DataFrame): The dataset to be split.
        target_col (str): The name of the target column in the dataset.
        num_folds (int): Number of folds. Default is 5.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) dataframe 
                                                 tuples for each fold.
    Raises:
        ValueError: If a test fold does not contain samples from all classes.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
    
    fold_datasets = []
    unique_classes = dataset[target_col].unique()
    
    for train_index, test_index in skf.split(dataset, dataset[target_col]):
        train_fold = dataset.iloc[train_index]
        test_fold = dataset.iloc[test_index]
        # Check if all classes are present in the test fold
        test_classes = test_fold[target_col].unique()
        if not all(cls in test_classes for cls in unique_classes):
            missing_classes = set(unique_classes) - set(test_classes)
            raise ValueError(f"Test fold is missing samples from classes: {missing_classes}")

        fold_datasets.append((train_fold, test_fold))

    return fold_datasets

def save_train_data(
        train_df: pd.DataFrame, dataset_name: str, processed_datasets_path: str) -> None:
    """
    Saves the train data to a CSV file.

    Args:
        train_df (DataFrame): The train dataset.
        dataset_name (str): The name of the dataset.
        processed_datasets_path (str): The path where the processed datasets are stored.
    """
    train_df.to_csv(os.path.join(
        processed_datasets_path, dataset_name, f"{dataset_name}_train.csv"), index=False)

def save_test_no_target_data(
        test_df: pd.DataFrame,
        target_name: str,
        dataset_name: str,
        processed_datasets_path: str) -> None:
    """
    Saves the test data without the target column to a CSV file.

    Args:
        test_df (DataFrame): The test dataset.
        target_name (str): The name of the target column.
        dataset_name (str): The name of the dataset.
        processed_datasets_path (str): The path where the processed datasets are stored.
    """
    test_no_target_df = test_df.drop(target_name, axis=1)
    test_no_target_df.to_csv(
        os.path.join(processed_datasets_path, dataset_name, f"{dataset_name}_test.csv"), index=False)

def save_test_key_data(
        test_df: pd.DataFrame,
        id_name: str,
        target_name: str,
        dataset_name: str,
        processed_datasets_path: str) -> None:
    """
    Saves the test key data to a CSV file.

    Args:
        test_df (DataFrame): The test dataset.
        id_name (str): The name of the ID column.
        target_name (str): The name of the target column.
        dataset_name (str): The name of the dataset.
        processed_datasets_path (str): The path where the processed datasets are stored.
    """
    test_key_df = test_df[[id_name, target_name]]
    test_key_df.to_csv(
        os.path.join(processed_datasets_path, dataset_name, f"{dataset_name}_test_key.csv"), index=False)


def save_train_test_testkey_files(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
        id_name: str,
        target_name: str,
        processed_datasets_path=paths.processed_datasets_path
    ):
    """Saves train, test, and test key files for each dataset marked for use in the metadata."""
    # Save train/test data
    save_train_data(train_df, dataset_name, processed_datasets_path)

    # Save test data without target
    save_test_no_target_data(
        test_df, target_name, dataset_name, processed_datasets_path)

    # Save test key data
    save_test_key_data(
        test_df, id_name, target_name, dataset_name, processed_datasets_path)
