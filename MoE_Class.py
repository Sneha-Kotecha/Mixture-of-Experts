import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertModel, BertConfig
from load_medqa_data import load_med_qa_data
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch.nn.functional as F

from Config import Config

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MoE(nn.Module):
    def __init__(self, num_experts, hidden_size, output_size):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(Config.dropout_rate)

    def forward(self, inputs):
        batch_size, seq_length, hidden_size = inputs.size()
        inputs = inputs.view(batch_size * seq_length, hidden_size)
        gate_outputs = self.gate(self.dropout(inputs))
        gate_outputs = F.softmax(gate_outputs, dim=1)
        expert_outputs = [expert(self.dropout(inputs)) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = (expert_outputs * gate_outputs.unsqueeze(2)).sum(dim=1)
        expert_outputs = expert_outputs.view(batch_size, seq_length, -1)
        return expert_outputs