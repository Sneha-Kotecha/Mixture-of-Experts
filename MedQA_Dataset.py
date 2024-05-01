import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertModel, BertConfig
from load_medqa_data import load_med_qa_data
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch.nn.functional as F

class MedQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]

        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert answer index (e.g., "E") to label (0, 1, 2, 3, 4)
        label = list(options.keys()).index(answer_idx)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label": torch.tensor(label),
            "options": options,
        }