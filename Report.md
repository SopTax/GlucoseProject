
# Assignment: Time Series Forecasting of Blood Glucose Levels for Type 1 Diabetes Management
## 1. The Data
The OhioT1DM dataset, which includes continuous glucose monitoring data, insulin dosages, and self-monitored blood glucose values among other variables, serves as the foundation for this study.
## 2. Methodology
I approached the problem in an iterative way. 
Firstly, I drafted the scheleton of the key elements: a loader/preprocessor, the model and a basic training loop.
Then, iteratively refined the various building blocks and adding operations and complexity.

### 2.1 Data Preprocessing
The three main aspects I encountered then it comes to the preprocessing are:
1. Missing data: how to address NaNs, which imputation methods to use.
    a. For the target variable, cbg, the imputation was done with Cubic Spline interpolation (coherent with the method used in the paper DOI: https://doi.org/10.48550/arXiv.2109.02178 )
    b. For all other variables, the NaNs were filled with a value equal to the minimum of that feature minus a small percentile of this minimum. This is done with in mind the successive normalization step to (0,1), in order to not set those values to 0 or -1 as often is done, and allowing for the nn to (hopefully) recognize them as 'strange'.
3. Merging: making sure the correct features are kept, in the right order for the model to then predict the desired variable; also mind that the data is ordered from a temporal point of view.
4. Scaling: the normalization step was important as it requires also the inclusion of an un-scaling step.

### 2.2 Model Architecture
In our examination of related work, we look at the study by Mirshekarian et al., which focuses on leveraging a Long Short-Term Memory (LSTM) neural network for predicting blood glucose levels. Their model architecture is relatively simple and efficient, comprising a single hidden LSTM layer with five units. This layer is succeeded by a dense output layer that has a singular neuron. The selected activation functions are sigmoid for the LSTM's recurrent computations and tanh for the transformations within the LSTM layer. The output neuron uses a linear activation function, which is apt for regression tasks such as blood glucose prediction. The model takes input with four features across 25 time steps, reflecting the time series nature of the dataset used for training and evaluation.
I then added Batch Normalization.

### 2.3 Training Process
Coherently with the Tena, Felix, et al. paper, the training was initially done over 100 epochs, with a learning rate of 0.01 and input sequence length of 24 time steps (120 minutes).
Successively, during the fine-tuning process, the learning rate was set to 0.001 initially and is then divided by 10 after the 80th epoch.
In the final setting, the batch size was set to 500.

## 3. Results
### 3.1 Training and Validation Losses
(Show graphs of training and validation losses over the epochs to illustrate the learning process.)

### 3.2 Prediction Performance
(Include visualizations of the model's predictions against the actual values and discuss the accuracy.)

### 3.3 Metrics
(Report the average metrics and standard error of the mean for model evaluation.)
