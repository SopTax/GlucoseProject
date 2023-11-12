import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Import necessary modules from other scripts
from data_processor_loader import create_dataloader, preprocess, get_preprocessor, get_scaler
from cnn_model import SimpleCNN
from training_function import train, plot_losses

def get_prediction_ahead(model: t.nn.Module, input_seq:t.Tensor, n_steps:int):
  model.eval()
  with t.no_grad():
    cur_input_seq = input_seq.clone()
    for i in range(n_steps):
      next_value = model(cur_input_seq)[:, None]
      cur_input_seq = t.cat([cur_input_seq[:, 1:], next_value], dim=1)
  model.train()
  return cur_input_seq[:, -n_steps:]


def plot_predictions(model: nn.Module,n_steps_ahead:int=6):
  train_seq_len=24
  dataloader = create_dataloader(
      data_dirs=test_data_dirs,
      seq_length=train_seq_len + 1 + n_steps_ahead,  # Example sequence length
      batch_size=1
  )
  sample = next(iter(dataloader))
  sample_context = sample[0][:, :24]
  sample_ahead = sample[0][:, 24:]
  print(f"{sample_context.shape=}")
  print(f"{sample_ahead.shape=}")
  prediction = get_prediction_ahead(model, sample_context, n_steps_ahead)
  print(prediction.shape)
  prediction = prediction
  print(f"{prediction.shape=}")

  plt.clf()
  plt.close()
  total_timesteps = t.arange(train_seq_len + n_steps_ahead)
  plt.plot(total_timesteps[:train_seq_len], dataloader.unscale(sample_context).squeeze()[:, -1], label='context')
  plt.plot(total_timesteps[-n_steps_ahead:], dataloader.unscale(sample_ahead).squeeze()[:, -1], label='true')
  plt.plot(total_timesteps[-n_steps_ahead:], dataloader.unscale(prediction).squeeze()[:, -1], label='pred')
  plt.legend()
  plt.show()

def compute_metrics(model: nn.Module):
    n_steps_ahead = 6
    train_seq_len = 24

    dataloader = create_dataloader(
        data_dirs=test_data_dirs,
        seq_length=train_seq_len + 1 + n_steps_ahead,  # Example sequence length
        batch_size=1
    )

    # Initialize metrics
    mse_values = []
    mae_values = []
    rmse_values = []
    r2_values = []
    cc_values = []  # Correlation Coefficient
    fit_values = []  # Fit Index
    mard_values = []  # Mean Absolute Relative Difference

    for sample in dataloader:
        sample_context = sample[0][:, :24]
        sample_ahead = sample[0][:, 24:]

        prediction = get_prediction_ahead(model, sample_context, n_steps_ahead)
        prediction = dataloader.unscale(prediction).squeeze()[:, -1]
        sample_ahead = dataloader.unscale(sample_ahead).squeeze()[:, -1]

        # Update metrics
        mse = mean_squared_error(sample_ahead, prediction)
        mse_values.append(mse)

        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

        mae = mean_absolute_error(sample_ahead, prediction)
        mae_values.append(mae)

        #r2 = r2_score(sample_ahead, prediction)
        #r2_values.append(r2)

        #cc = np.corrcoef(sample_ahead, prediction)
        #cc_values.append(cc[0, 1])


        #numerator =  (abs(prediction - sample_ahead.mean()).sum())/len(sample_ahead)
        #numerator = (t.abs(prediction - t.mean(sample_ahead)).sum())/len(sample_ahead)  # Sum of squares of differences
        #denominator = (abs(sample_ahead - sample_ahead.mean()).sum())/len(sample_ahead)
        #denominator = (t.abs(sample_ahead - t.mean(sample_ahead)).sum())/len(sample_ahead)  # Total sum of squares
        #fit = 1 - (numerator / denominator) if denominator != 0 else 0  # Handle division by zero
        #fit_values.append(fit)

        relative_diff = (sample_ahead - prediction)
        mard = t.mean(t.abs(relative_diff)/sample_ahead)
        mard_values.append(mard)

    # Aggregate metrics
    metrics = {
        'MSE': np.mean(mse_values),
        'MSE_sd': np.std(mse_values),
        'RMSE': np.mean(rmse_values),
        'RMSE_sd': np.std(rmse_values),
        'MAE': np.mean(mae_values),
        'MAE_sd': np.std(mae_values),
        #'R2': np.mean(r2_values),
        #'CC': np.mean(cc_values),
        #'FIT': np.mean(fit_values),
        'MARD': np.mean(mard_values),
        'MARD_sd': np.std(mard_values),
    }

    return metrics



def main():
    # Define model parameters
    input_size = 7
    hidden_size = 5
    num_layers = 1
    output_size = 7
    lr = 0.001
    batch_size = 500
    num_epochs = 150

    # Initialize models
    lstm_model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)

    # Train the model
    train_losses, test_losses, trained_lstm_model = train(
        net_class=SimpleLSTM,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs
    )

    # Plot predictions
    plot_predictions(trained_lstm_model, 6)
    plot_predictions(trained_lstm_model, 12)
    plot_predictions(trained_lstm_model, 24)

    # Compute and print metrics
    metrics = compute_metrics(trained_lstm_model)
    print(metrics)

if __name__ == "__main__":

    main()
