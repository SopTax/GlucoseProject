import os
import zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader
import torch as t

path=os.getcwd()



class OhioT1DMDataset(Dataset):
    def __init__(self, data_dirs, seq_length):
        self.seq_length = seq_length

        # Initialize an empty DataFrame
        merged_data = pd.DataFrame()
        dataframes = []
        # Load and merge data from all directories provided
        for data_dir in data_dirs:
            for subdir, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    # Assuming the data is in a CSV format
                    data_df = pd.read_csv(file_path)
                    # Append the data to the merged_data DataFrame
                    merged_data = pd.concat([merged_data, data_df])
                    dataframes.append(data_df)

        merged_data.reset_index(inplace=True)
        # Apply the preprocess function to the merged data
        scaler, fill_values = get_scaler(merged_data)
        self.scaler = scaler
        self.preprocessed_dfs = [preprocess(scaler, fill_values, df) for df in dataframes]

        # Convert the DataFrame to a PyTorch tensor
        self.data = [t.tensor(df.values, dtype=t.float32) for df in self.preprocessed_dfs]

    def __len__(self):
        # Return the total number of sequences available
        return sum(len(data) - self.seq_length + 1 for data in self.data)

    def __getitem__(self, index):
        # Find which data point this index is referring to
        data_idx = 0
        while index >= len(self.data[data_idx]) - self.seq_length + 1:
            index -= len(self.data[data_idx]) - self.seq_length + 1
            data_idx += 1

        # Extract the sequence
        sequence = self.data[data_idx][index:index+self.seq_length]
        # Split the sequence into inputs and target
        inputs = sequence[:-1,:]  # All but the last element
        target = sequence[-1,:]  # Only the last element

        return inputs, target
# Adjust the create_dataloader function to accept a list of directories
def create_dataloader(data_dirs, seq_length, batch_size):
    # Instantiate the custom dataset
    dataset = OhioT1DMDataset(data_dirs, seq_length)
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True);

    def unscale(data):
      data = data.squeeze(0)
      df = pd.DataFrame(data, columns=dataset.preprocessed_dfs[0].columns)
      unscaled = dataset.scaler.inverse_transform(df)
      return t.tensor(unscaled)[None]

    dataloader.__dict__['unscale'] = unscale
    return dataloader

# Example usage of the function to create a DataLoader for the combined train data
train_data_dirs = [
    f"{path}/Ohio Data/Ohio2018_processed/train",
    f"{path}/Ohio Data/Ohio2020_processed/train"
]
test_data_dirs = [
    f"{path}/Ohio Data/Ohio2018_processed/test",
    f"{path}/Ohio Data/Ohio2020_processed/test"
]


def preprocess(scaler, fill_values, data_df):
  data_df1 = data_df.copy()
  # Identify rows where 'missing_cbg' is 1
  missing_cbg_indices = data_df1[data_df1['missing_cbg'] == 1].index

  # Perform cubic spline interpolation for 'cbg' column
  cs = CubicSpline(
    data_df1.index[~data_df1.index.isin(missing_cbg_indices)],
    data_df1.loc[~data_df1.index.isin(missing_cbg_indices), 'cbg']
  )

  data_df1.loc[missing_cbg_indices, 'cbg'] = cs(missing_cbg_indices)

  data_df1 = data_df1.drop(columns=[
      '5minute_intervals_timestamp',
      'missing_cbg'
    ]
  )

  # Move 'cbg' to the end
  cbg = data_df1.pop('cbg')
  data_df1 = data_df1.assign(cbg=cbg)

  column_mins = data_df1.min()
  # Subtract a small percentile of the minimum from the minimum
  values = data_df1.values
  """
  print("## VALUES BEFORE")
  print(values)
  """
  values = np.where(np.isnan(values), fill_values, values)
  """
  print("## FILL VALUES")
  print(fill_values)
  print(fill_values.shape)
  print("## VALUES")
  print(values)
  """


  data_df2 = pd.DataFrame(values, columns=data_df1.columns)
  data_df2 = pd.DataFrame(scaler.transform(data_df2), columns=data_df2.columns)

  return data_df2


def get_scaler(data_df):
    data_df1 = data_df.copy()

    # Identify rows where 'missing_cbg' is 1
    missing_cbg_indices = data_df1[data_df1['missing_cbg'] == 1].index

    # Perform cubic spline interpolation for 'cbg' column
    cs = CubicSpline(
      data_df1.index[~data_df1.index.isin(missing_cbg_indices)],
      data_df1.loc[~data_df1.index.isin(missing_cbg_indices), 'cbg']
    )

    data_df1.loc[missing_cbg_indices, 'cbg'] = cs(missing_cbg_indices)

    # Move 'cbg' to the end
    cbg = data_df1.pop('cbg')
    data_df1 = data_df1.assign(cbg=cbg)


    # Drop time column
    data_df1 = data_df1.drop(columns=['5minute_intervals_timestamp', 'missing_cbg', 'index'])

    # Calculate the minimum of each column
    column_mins = data_df1.min()
    # Subtract a small percentile of the minimum from the minimum
    fill_values = column_mins - 0.01 * np.abs(column_mins)
    # Fill missing values with the calculated values
    data_df2 = data_df1.fillna(fill_values)

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()  # default=(0, 1)
    scaler.fit(data_df2)

    assert np.isnan(fill_values.values).sum() == 0, 'oh nooo'

    return scaler, fill_values
