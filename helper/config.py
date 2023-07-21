
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Choosen device { DEVICE }")
__days__, __hour__ = 7, 24
WINDOW_SIZE = __days__ * __hour__  # Three Day
COLUMN_NAMES = ['spring', 'summer', 'autumn', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'business_hour', 'weekend', 'daylight', 'holiday', 'temperature', 'demand']
GLOBAL_COLUMN_NAMES = ['bergen', 'helsingfors', 'oslo', 'stavanger', 'tromsÃ¸', 'spring', 'summer', 'autumn', 'month_sin', 'month_cos',
                'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'business_hour', 'weekend', 'daylight', 'holiday', 'temperature', 'demand']


# Input neurons size
# INPUT_SIZE = 16

# Hidden layer size
HIDDEN_SIZE = 100

# Output neuron size
OUTPUT_SIZE = 1

# Number of epochs, to train the model
NUM_EPOCHS = 10

# Numbers of epoch to go through the whole data set in each epoch
BATCH_SIZE = 20

# Percentage of training set per epoch
PRC_EPOCH = 1

# Number of delay before predicting the first value
PREDICT_DELAY = 3  # 3 hours
