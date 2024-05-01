import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Config:
    max_length = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-5
    batch_size = 8
    num_epochs = 10  # Increased the number of epochs
    num_experts = 5
    weight_decay = 0.01  # Add weight decay for regularization
    dropout_rate = 0.2 # Added dropout rate
    num_choices = 5  
