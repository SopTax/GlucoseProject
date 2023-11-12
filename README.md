# Glucose Level Prediction with LSTM
This repository contains a collection of scripts used to predict glucose levels using LSTM and CNN models. The data processing, model definitions, training, and evaluation are split across multiple scripts for modularity and ease of use.
## Project Details
Time series forecasting for blood glucose (BG) values.
### Data
Data recorded from 12 different PwT1D (Type I diabetes patients). 
The data was released in the OhiaT1DM dataset (Marling, C., & Bunescu, R. (2020). The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020).
The datasets include information such as continuous glucose monitoring (CGM), BG values obtained through self-monitoring by the patient (finger stick), basal insulin rate, bolus injection, the self-reported time and type of a meal, plus the patient’s carbohydrate estimate for the meal. The timestamps are over intervals of 5 minutes (so, for example, 24 timesteps are equal to ain interval of 120 minutes).

### Scope of the project
Develop a patient-personalized forcasting model for blood glucose value.

## The Repo
### Structure
- **data_processor_loader.py**: Handles data preprocessing, loading, and the creation of PyTorch datasets.
- **lstm_model.py**: Defines the architecture of the LSTM model used for time-series prediction.
- **training_function.py**: Includes the training loop and loss visualization functions.
- **main.py**: The main script that orchestrates the training process, generates predictions, plots results, and calculates performance metrics.
### Setup
To run these scripts, you will need to have Python 3.6 or later installed along with the following packages:
- pandas
- numpy
- pytorch
- matplotlib
- sklearn
The libraries versions for recreation of the environment are in the Gluc_proj_env.yaml file.
Before running the scripts, ensure that your dataset is located in the **'Ohio Data'** directory within the root of this project. If your data is in a different location, update the paths accordingly in the **data_processor_loader.py** script. The structure of the **'Ohio Data'** folder is as follows: 
```bash
Ohio Data/
│
├── Ohio2018_preprocessed/
│   ├── train/
│   │   ├── ...
│   │
│   └── test/
│       ├── ...
└── Ohio2020_preprocessed/
    ├── train/
    │   ├── ...
    └── test/
        ├── ...
```
### Usage
To run the experiment:
1. Ensure that your data is placed in the appropriate directory specified in data_processor_loader.py.
2. Configure your model and training parameters in main.py.
3. Execute the main script:
```bash
python main.py
```
### Visualization
The main.py script will produce plots for the training and testing losses, as well as prediction visualizations and metrics after the training process.

## Notebook
Additionally, the Colab notebook  where the project was originally created and carried out is included. 
Here some additional functions are included. The path needs to be adjust to run the notebook, to be the path to the directory with the Ohio Data folder.

