import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertModel, BertConfig
from load_medqa_data import load_med_qa_data
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch.nn.functional as F
from preprocess_data import preprocess_data
from train import train
from evaluate import evaluate

from Config import Config
from MedQA_Dataset import MedQADataset
from MultipleChoiceModel import MultipleChoiceModel


def main(train_file, val_file, test_file):
    train_data, val_data, test_data = preprocess_data(train_file, val_file, test_file)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = MedQADataset(train_data, tokenizer, Config.max_length)
    val_dataset = MedQADataset(val_data, tokenizer, Config.max_length)
    test_dataset = MedQADataset(test_data, tokenizer, Config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = MultipleChoiceModel('bert-base-uncased', num_choices=5, num_experts=Config.num_experts, tokenizer=tokenizer).to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    train(model, train_loader, val_loader, optimizer, criterion, Config.num_epochs, scheduler)

    test_metrics = evaluate(model, test_loader)
    test_f1_score = test_metrics['f1_score']
    test_class_report = test_metrics['classification_report']

    print(f"Test F1-Score: {test_f1_score:.4f}")
    print("Test Classification Report:")
    print(test_class_report)

if __name__ == "__main__":
    train_file = '/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/train.jsonl'
    val_file = '/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/dev.jsonl'
    test_file = '/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/test.jsonl'
    main(train_file, val_file, test_file)