import pandas as pd
from typing import Dict, List


def create_feature_section(
        dataset_name: str,
        dataset_row: pd.Series,
        dataset: pd.DataFrame,
        features_config: pd.DataFrame) -> List[Dict]:
    """
    Create the feature section of the schema.

    Args:
    dataset_name (str): The name of the dataset.
    dataset_row (pd.Series): The metadata for the dataset.
    dataset (pd.DataFrame): The dataset.
    features_config (pd.DataFrame): The features configuration data.

    Returns:
    List[Dict]: The features section of the schema.
    """
    # Filter features related to this dataset
    features_config = features_config[features_config['name'] == dataset_row['name']]

    # create the features section
    features = []
    features_df = features_config[(features_config["name"]==dataset_name)
                                  & (features_config["field_type"]=="feature")]
    
    for _, feature_row in features_df.iterrows():
        feature = {
            "name": feature_row['field_name'],
            "description": feature_row['field_description'],
            "dataType": feature_row['data_type'].upper(),
        }
        if feature_row['data_type'].upper() == "CATEGORICAL":
            feature["categories"] = sorted(dataset[feature_row['field_name']]\
                                           .dropna().unique().tolist(), key=str)
        else:
            feature["example"] = dataset[feature_row['field_name']].dropna().iloc[0]
        feature["nullable"] = dataset[feature_row['field_name']].isnull().any()
        features.append(feature)

    return features


def generate_schema_for_dataset(
        dataset_cfg: pd.Series,
        features_config: pd.DataFrame,
        dataset: pd.DataFrame
    ) -> Dict:
    """
    Generate the schema for each dataset.

    Args:
    dataset_cfg (pd.Series): The metadata for all the datasets.
    features_config (pd.DataFrame): The features configuration data.
    dataset (pd.DataFrame): The main dataset.
    """

    dataset_name = dataset_cfg["name"]
    schema = {}
    schema["title"] = dataset_cfg["title"]
    schema["description"] = dataset_cfg["description"]
    schema["modelCategory"] = dataset_cfg["model_category"]
    schema["schemaVersion"] = 1.0
    schema["inputDataFormat"] = "CSV"
    schema["encoding"] = dataset_cfg["encoding"]

    schema["id"] = {
        "name": dataset_cfg["id_name"],
        "description": dataset_cfg["id_description"]
    }

    schema["target"] = {
        "name": dataset_cfg["target_name"],
        "description": dataset_cfg["target_description"],
        "classes": dataset_cfg["target_classes"].split('|')
    }

    schema["features"] = create_feature_section(
        dataset_name, dataset_cfg, dataset, features_config)

    return schema
