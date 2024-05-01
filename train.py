import torch
import numpy as np
import torch.nn.functional as F
from Config import Config
from evaluate import evaluate

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, scheduler):
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            token_type_ids = batch["token_type_ids"].to(Config.device)
            labels = batch["label"].to(Config.device)
            options = batch["options"]

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, token_type_ids, options)
            labels_one_hot = F.one_hot(labels, num_classes=Config.num_choices).long()
            loss = criterion(outputs, labels_one_hot)

            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, criterion)
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1_score']
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation F1-Score: {val_f1:.4f}")

        # Update learning rate scheduler with validation loss
        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    print(f"Best Validation F1-Score: {best_val_f1:.4f}")
