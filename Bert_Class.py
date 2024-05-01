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

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(Config.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        return sequence_output