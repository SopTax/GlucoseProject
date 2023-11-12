import torch as t
from torch import nn

import torch.optim as optim
import matplotlib.pyplot as plt
from data_processor_loader import create_dataloader
import os
path=os.getcwd()
train_data_dirs = [
    f"{path}/Ohio Data/Ohio2018_processed/train",
    f"{path}/Ohio Data/Ohio2020_processed/train"
]
test_data_dirs = [
    f"{path}/Ohio Data/Ohio2018_processed/test",
    f"{path}/Ohio Data/Ohio2020_processed/test"
]

# Include the training function
def train(
  net_class:type,
  input_size:int,
  hidden_size:int,
  num_layers:int,
  output_size:int,
  lr:float = 0.01,
  num_epochs:int=100,
  batch_size:int=256
):
  # We use a default batch size of 64 as an example
  train_dataloader = create_dataloader(
      data_dirs=train_data_dirs,
      seq_length=25,  # Example sequence length
      batch_size=batch_size
  )
  test_dataloader = create_dataloader(
      data_dirs=train_data_dirs,
      seq_length=25,  # Example sequence length
      batch_size=batch_size
  )


  model = net_class(input_size, hidden_size, num_layers, output_size)
  print(model)

  # Define loss function and optimizer
  criterion = nn.MSELoss()  # for regression tasks; use nn.CrossEntropyLoss for classification tasks
  optimizer = optim.Adam(model.parameters(), lr=lr)

  train_losses: list[float] = []
  test_losses: list[float] = []

  best_validation_loss = float('inf')
  best_model_dict = None

  # Training loop
  for epoch in range(num_epochs):

      epoch_train_losses = []
      model.train()  # Set model to training mode
      for inputs, targets in train_dataloader:
          # Zero the parameter gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          # Backward pass and optimize
          loss.backward()
          optimizer.step()
          epoch_train_losses.append(loss)

      if epoch == 80:
        lr /= 10

      # Validation phase
      model.eval()  # Set model to evaluation mode
      epoch_test_losses = []
      with t.no_grad():  # No gradients required for validation
          for inputs, targets in test_dataloader:
              outputs = model(inputs)
              # print(f'{outputs.shape=}')
              # print(f'{targets.shape=}')
              loss = criterion(outputs[:,-1], targets[:, -1])
              epoch_test_losses.append(loss.item())

      # Calculate average validation loss
      # validation_loss /= len(validation_data_loader)
      # Print statistics
      #print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {validation_loss:.4f}')
      mean_train_loss = t.mean(t.tensor(epoch_train_losses))
      mean_test_loss = t.mean(t.tensor(epoch_test_losses))
      print(f'Epoch {epoch+1}/{num_epochs}\tTrain MSE: {mean_train_loss.item():.9f}\tTest MSE {mean_test_loss.item():.9f}')
      # Store the loss value
      train_losses.append(mean_train_loss.item())
      test_losses.append(mean_test_loss.item())

      current_validation_loss = mean_test_loss.item()
      if current_validation_loss < best_validation_loss:
          best_validation_loss = current_validation_loss
          best_model_dict = model.state_dict()  # Save the model parameters


  # Plot the loss values
  plot_losses(train_losses, test_losses)
  t.save(best_model_dict, 'best_model.pth')

  assert best_model_dict is not None
  model.load_state_dict(best_model_dict)

  t.save(lstm_model.state_dict(), 'simple_lstm_model.pth')

  return train_losses, test_losses, model

# Include the plot_losses function
def plot_losses(train_losses, test_losses):
    # ... [plot_losses function definition] ...
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Plot for train losses
    axs[0].plot(t.arange(len(train_losses[20:])), train_losses[20:], label='train')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Train Loss')

    # Plot for test losses
    axs[1].plot(t.arange(len(test_losses[20:])), test_losses[20:], label='test')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title('Test Loss')

    # Set common x label
    plt.xlabel('Epoch')
    plt.show()
