import torch.nn as nn
import torch as t

class SimpleLSTM(nn.Module):
    # ... [SimpleLSTM class definition] ...
  def __init__(
      self,
      input_size: int,
      hidden_size:int,
      num_layers:int,
      output_size:int
    ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_norm = nn.BatchNorm1d(input_size)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Initialize hidden and cell states
    h0 = t.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = t.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])
    return out
