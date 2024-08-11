# SMOTE Analysis datasets

This repo contains files related to the datasets used for the smote analysis project. There are a total of 31 benchmarking datasets used in this project. The list of datasets is as follows:
| Dataset | # of features | # of samples | % of minority class | source |
|-------------------------------------------------------|:---------------------------:|:----------------:|:----------------:|:------------------------------:|
| abalone_binarized | 9 | 4177 | 9.3% | https://imbalanced-learn.org/stable/datasets/index.html#imbalanced-datasets-for-benchmark |
| auction | 7 | 2043 | 12.8% | https://archive.ics.uci.edu/dataset/713/auction+verification |
| car_eval_binarized | 21 | 1728 | 7.7% | https://imbalanced-learn.org/stable/datasets/index.html#imbalanced-datasets-for-benchmark |
| chess | 6 | 2901 | 3.6% | https://sci2s.ugr.es/keel/dataset.php?cod=1334 |
| climate_simulation_crashes | 20 | 540 | 8.5% | https://www.openml.org/search?type=data&status=any&id=1467 |
| club_loan | 13 | 9578 | 16% | https://www.kaggle.com/datasets/swetashetye/lending-club-loan-data-imbalance-dataset |
| coil_2000 | 85 | 9822 | 5.9% | https://imbalanced-learn.org/stable/datasets/index.html |
| graduation | 4 | 1687 | 8% | https://www.kaggle.com/datasets/oddyvirgantara/on-time-graduation-classification |
| jm1 | 21 | 10885 | 19.3% | https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=%3D_2&id=1053 |
| kc1 | 21 | 2109 | 15.4% | https://www.openml.org/search?type=data&status=active&id=1067 |
| letter_img | 16 | 20000 | 3.6% | https://imbalanced-learn.org/stable/datasets/index.html |
| mammography | 6 | 11183 | 3.6% | https://imbalanced-learn.org/stable/datasets/index.html |
| optical_digits | 64 | 5620 | 9.8% | https://imbalanced-learn.org/stable/datasets/index.html |
| ozone_level | 72 | 2536 | 6.3% | https://imbalanced-learn.org/stable/datasets/index.html |
| page_blocks | 10 | 5472 | 10.2% | https://archive.ics.uci.edu/dataset/78/page+blocks+classification |
| pc1 | 21 | 1109 | 6.9% | https://www.openml.org/search?type=data&status=any&id=1068 |
| pen_digits | 16 | 10,992 | 9.5% | https://imbalanced-learn.org/stable/datasets/index.html |
| pie_chart | 37 | 1077 | 12.4% | https://www.openml.org/search?type=data&status=active&id=1453 |
| satellite | 36 | 5100 | 1.4% | https://www.openml.org/search?type=data&status=active&id=40900 |
| satimage | 36 | 6435 | 9.7% | https://imbalanced-learn.org/stable/datasets/index.html |
| seismic_bumps | 15 | 2584 | 6.5% | https://www.openml.org/search?type=data&status=active&id=45562 |
| shuttle | 9 | 1829 | 6.7% | https://sci2s.ugr.es/keel/dataset.php?cod=125|
| sick_euthyroid | 42 | 3163 | 9.2% | https://imbalanced-learn.org/stable/datasets/index.html |
| solar_flare | 32 | 1389 | 4.9% | https://imbalanced-learn.org/stable/datasets/index.html |
| thoracic_surgery | 16 | 470 | 14.8% | https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data |
| thyroid_disease | 27 | 3772 | 6.1% | https://imbalanced-learn.org/stable/datasets/index.html |
| us_crime | 100 | 1994 | 7.5% | https://imbalanced-learn.org/stable/datasets/index.html |
| vowel | 13 | 988 | 9.1% | https://sci2s.ugr.es/keel/dataset.php?cod=127 |
| wilt | 5 | 4839 | 5.3% | https://www.openml.org/search?type=data&status=active&id=40983 |
| wine_quality | 11 | 4898 | 3.7% | https://imbalanced-learn.org/stable/datasets/index.html |
| yeast | 8 | 1484 | 10.9% | https://sci2s.ugr.es/keel/dataset.php?cod=154 |

## Repository Structure

The `datasets` folder contains the main data files and the schema files for all the benchmark datasets under Binary Classification category.

- `processed` folder contains the processed files. These files are used in algorithm evaluations.

  - The CSV file with suffix "\_train.csv" is used for training
  - "\_test.csv" is used for testing (without the targets)
  - "\_test_key.csv" contains the ids and targets for the test data. This test key file is used to generate scores by comparing with predictions.
  - The JSON file with suffix "\_schema.json" is the schema file for the corresponding dataset.
  - The json file with the suffix "\_inference_requeest_sample.json" contains a sample JSON object with the data to make an inference request to the /infer endpoint.

- The `raw` folder contains the original data files from the source (see source urls in table above).

- The folder `.src/config` contains two csv files - one called `binary_classification_datasets_metadata.csv` which contains the dataset level attribute information. The second csv called `binary_classification_datasets_fields.csv` contains information regarding all the fields in each of the datasets.
- `raw_datasets_processing.py`: contains the code to read and preprocess the original source data into the required pandas dataframe format.
- `schema_gen.py`: contains the code to generate the schema files for each dataset.
- `train_test_key_files_gen.py`: contains the code to save the train, test, and test-key files for each dataset.
- `run_all.py`: This is used to run the above three scripts in sequence.

Note again that the main files for all the datasets are in the `./datasets/processed` folder.
