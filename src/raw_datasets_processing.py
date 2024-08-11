import os
import numpy as np
import pandas as pd
import ast

import paths


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all cell values in the dataset.

    Args:
        data (pd.DataFrame): The dataset to clean.

    Returns:
        pd.DataFrame: The dataset with stripped whitespace.
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
    return data


def convert_numeric_columns_to_feature_strings(data: pd.DataFrame) -> pd.DataFrame:
    """Convert columns named as integers (0, 1, 2, ...) to feature strings (f0, f1, ...).

    Args:
        data (pd.DataFrame): The dataset with columns to convert.

    Returns:
        pd.DataFrame: The dataset with converted column names.
    """
    new_column_names = []

    for col in data.columns:
        if col.isdigit():  # Check if the column name is numeric
            new_column_names.append(f'f{col}')
        else:
            new_column_names.append(col)

    data.columns = new_column_names
    return data



def insert_id_col_if_not_exists(data: pd.DataFrame, id_field: str)->pd.DataFrame:
    """Insert ID column if it does not exist

    Args:
        data (pd.DataFrame): The dataset.
        id_field (str): The name of the ID field.

    Returns:
        pd.DataFrame: The dataset with the ID field.
    """
    if id_field not in data.columns:
        data.insert(0, id_field, np.arange(len(data)))
    return data

def convert_byte_string_repr(entry):
    try:
        # Check if the entry looks like a byte string representation
        if isinstance(entry, str) and entry.startswith("b'") and entry.endswith("'"):
            byte_value = ast.literal_eval(entry)
            return byte_value.decode('utf-8')
    except (ValueError, SyntaxError):
        pass
    return entry  # Return the original entry if conversion fails


def convert_byte_strings_to_strings(data: pd.DataFrame)->pd.DataFrame:
    """Convert byte strings to strings

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with byte strings converted to strings.
    """
    byte_string_columns = list(data.select_dtypes(include=['O']).columns)
    for col in byte_string_columns:
        data[col] = data[col].apply(convert_byte_string_repr).astype(str)
    return data

def drop_single_value_columns(data: pd.DataFrame)->pd.DataFrame:
    """Drop columns that have only one unique value

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with the useless columns dropped.
    """
    unique_columns = [col for col in data.columns if data[col].nunique() == 1]
    data.drop(columns=unique_columns, inplace=True)
    return data

def replace_custom_nan_values(data: pd.DataFrame, nan_id: str = "?") -> pd.DataFrame:
    """Replace custom NaN values in the dataset with actual NaN values.

    Args:
        data (pd.DataFrame): The dataset in which to replace custom NaN values.
        nan_id (str, optional): The identifier for custom NaN values. Defaults to "?".
    
    Returns:
        pd.DataFrame: The dataset with custom NaN values replaced by actual NaN values.
    """
    data.replace(nan_id, np.nan, inplace=True)
    return data

def get_main_dataset_df(dataset_cfg: pd.Series)->pd.DataFrame:
    """Read and preprocess dataset

    Args:
        dataset_cfg (pd.Series): Metadata for the dataset, including name,
                                 id field, and target field.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    dataset_name = dataset_cfg['name']
    id_field = dataset_cfg['id_name']
    raw_fname = f"{dataset_name}_raw.csv"

    raw_data_fpath = os.path.join(
        paths.raw_datasets_path, dataset_name, raw_fname
    )
    data = pd.read_csv(raw_data_fpath)

    data = strip_whitespace(data)
    data = convert_numeric_columns_to_feature_strings(data)
    data = insert_id_col_if_not_exists(data, id_field)
    data = convert_byte_strings_to_strings(data)
    data = drop_single_value_columns(data)
    data = replace_custom_nan_values(data)
    return data