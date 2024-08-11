import os
import pandas as pd

from schema_gen import generate_schema_for_dataset
from utils import load_metadata, load_features_config, strip_quotes, save_json
from raw_datasets_processing import get_main_dataset_df
from train_test_key_files_gen import create_stratified_kfolds, \
    save_train_test_testkey_files
import paths


def process_all_datasets():
    dataset_metadata = load_metadata(paths.dataset_cfg_path)
    features_config = load_features_config(paths.features_cfg_path).apply(strip_quotes)

    num_folds = 5

    for _, dataset_row in dataset_metadata.iterrows():
        if dataset_row["use_dataset"] == 0:
            continue
        dataset_name = dataset_row["name"]
        id_field = dataset_row['id_name']
        target_field = dataset_row['target_name']
        print("Processing dataset:", dataset_name)

        main_dataset_df = get_main_dataset_df(dataset_cfg=dataset_row)

        schema = generate_schema_for_dataset(
            dataset_cfg=dataset_row,
            features_config=features_config,
            dataset=main_dataset_df
        )

        fold_datasets = create_stratified_kfolds(
            dataset=main_dataset_df,
            num_folds=num_folds,
            target_col=dataset_row["target_name"],
            random_state=123
        )

        for fold_num, fold_data in enumerate(fold_datasets):

            dataset_name_with_fold = f"{dataset_name}_fold_{fold_num}"
            train, test = fold_data

            # create processed dataset folder if not exists
            os.makedirs(
                os.path.join(paths.processed_datasets_path, dataset_name_with_fold),
                exist_ok=True
            )

            # Save schema
            save_json(
                schema,
                os.path.join(
                    paths.processed_datasets_path,
                    dataset_name_with_fold,
                    f"{dataset_name_with_fold}_schema.json"
                )
            )

            # Save train/test/test-key data
            save_train_test_testkey_files(
                train_df=train,
                test_df=test,
                dataset_name=dataset_name_with_fold,
                id_name=id_field,
                target_name=target_field
            )


if __name__ == "__main__":
    process_all_datasets()
