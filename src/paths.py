
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_cfg_path = os.path.join(ROOT_DIR, "src/config/smote_project_datasets.csv")
features_cfg_path = os.path.join(ROOT_DIR, "src/config/smote_project_datasets_fields.csv")
raw_datasets_path = os.path.join(ROOT_DIR, "datasets/raw/")
processed_datasets_path = os.path.join(ROOT_DIR, "datasets/processed/")

