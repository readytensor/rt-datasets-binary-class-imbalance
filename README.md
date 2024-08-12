# SMOTE Analysis Project Datasets

This repository contains files related to the datasets used for the SMOTE analysis project. There are a total of 30 benchmarking datasets used in this project. It creates 5-fold cross-validation sets for each of the 30 included datasets under the Binary Classification category.

## Datasets

The list of datasets is as follows:

| Dataset                    | # of features | # of samples | % of minority class |                                                source                                                 |
| -------------------------- | :-----------: | :----------: | :-----------------: | :---------------------------------------------------------------------------------------------------: |
| abalone_binarized          |       9       |     4177     |        9.3%         |   [link](https://imbalanced-learn.org/stable/datasets/index.html#imbalanced-datasets-for-benchmark)   |
| auction                    |       7       |     2043     |        12.8%        |                 [link](https://archive.ics.uci.edu/dataset/713/auction+verification)                  |
| car_eval_binarized         |      21       |     1728     |        7.7%         |   [link](https://imbalanced-learn.org/stable/datasets/index.html#imbalanced-datasets-for-benchmark)   |
| chess                      |       6       |     2901     |        3.6%         |                        [link](https://sci2s.ugr.es/keel/dataset.php?cod=1334)                         |
| climate_simulation_crashes |      20       |     540      |        8.5%         |                  [link](https://www.openml.org/search?type=data&status=any&id=1467)                   |
| club_loan                  |      13       |     9578     |         16%         |     [link](https://www.kaggle.com/datasets/swetashetye/lending-club-loan-data-imbalance-dataset)      |
| coil_2000                  |      85       |     9822     |        5.9%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| graduation                 |       4       |     1687     |         8%          |       [link](https://www.kaggle.com/datasets/oddyvirgantara/on-time-graduation-classification)        |
| jm1                        |      21       |    10885     |        19.3%        | [link](https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=%3D_2&id=1053) |
| kc1                        |      21       |     2109     |        15.4%        |                 [link](https://www.openml.org/search?type=data&status=active&id=1067)                 |
| letter_img                 |      16       |    20000     |        3.6%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| mammography                |       6       |    11183     |        3.6%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| optical_digits             |      64       |     5620     |        9.8%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| ozone_level                |      72       |     2536     |        6.3%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| page_blocks                |      10       |     5472     |        10.2%        |               [link](https://archive.ics.uci.edu/dataset/78/page+blocks+classification)               |
| pc1                        |      21       |     1109     |        6.9%         |                  [link](https://www.openml.org/search?type=data&status=any&id=1068)                   |
| pen_digits                 |      16       |    10,992    |        9.5%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| pie_chart                  |      37       |     1077     |        12.4%        |                 [link](https://www.openml.org/search?type=data&status=active&id=1453)                 |
| satellite                  |      36       |     5100     |        1.4%         |                [link](https://www.openml.org/search?type=data&status=active&id=40900)                 |
| satimage                   |      36       |     6435     |        9.7%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| seismic_bumps              |      15       |     2584     |        6.5%         |                [link](https://www.openml.org/search?type=data&status=active&id=45562)                 |
| sick_euthyroid             |      42       |     3163     |        9.2%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| solar_flare                |      32       |     1389     |        4.9%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| thoracic_surgery           |      16       |     470      |        14.8%        |                 [link](https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data)                 |
| thyroid_disease            |      27       |     3772     |        6.1%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| us_crime                   |      100      |     1994     |        7.5%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| vowel                      |      13       |     988      |        9.1%         |                         [link](https://sci2s.ugr.es/keel/dataset.php?cod=127)                         |
| wilt                       |       5       |     4839     |        5.3%         |                [link](https://www.openml.org/search?type=data&status=active&id=40983)                 |
| wine_quality               |      11       |     4898     |        3.7%         |                    [link](https://imbalanced-learn.org/stable/datasets/index.html)                    |
| yeast                      |       8       |     1484     |        10.9%        |                         [link](https://sci2s.ugr.es/keel/dataset.php?cod=154)                         |

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/smote-analysis-project-datasets.git
   cd smote-analysis-project-datasets
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run the following command to generate the processed files:

   ```bash
   python src/run_all.py
   ```

   This will create the processed files in the `datasets/processed` folder. There are 5-folds created for each of the 30 datasets, resulting in 150 folders in total. Each folder contains the train, test, and test-key files (which are CSVs) along with the schema file (JSON).

## Project Structure

The project is organized as follows:

- `datasets/`: Contains the main data files and schema files for all benchmark datasets under the Binary Classification category.

  - `processed/`: Contains the processed files used in algorithm evaluations.
    - Files with suffix "\_train.csv" are used for training.
    - Files with suffix "\_test.csv" are used for testing (without the targets).
    - Files with suffix "\_test_key.csv" contain the ids and targets for the test data, used to generate scores by comparing with predictions.
    - Files with suffix "\_schema.json" are the schema files for the corresponding datasets.
  - `raw/`: Contains the original data files from the source.

- `src/`: Contains the source code for processing the datasets.
  - `config/`: Contains two CSV files:
    - `binary_classification_datasets_metadata.csv`: Contains dataset-level attribute information.
    - `binary_classification_datasets_fields.csv`: Contains information about all fields in each dataset.
  - `raw_datasets_processing.py`: Code to read and preprocess the original source data into the required pandas dataframe format.
  - `schema_gen.py`: Code to generate the schema files for each dataset.
  - `train_test_key_files_gen.py`: Code to save the train, test, and test-key files for each dataset.
  - `run_all.py`: Main script that uses functions in the above three scripts to generate processed dataset files.

Note that the main files for all datasets are located in the `./datasets/processed` folder.

## License

The code in this repository is licensed under the MIT License. See the LICENSE file for details.

The datasets included in this repository are provided for convenience and are subject to their respective licenses as provided by the original authors and distributors. For more details on the licenses and to access the original datasets, please refer to the original sources listed in the table above.

Please ensure compliance with the respective licenses when using these datasets.
