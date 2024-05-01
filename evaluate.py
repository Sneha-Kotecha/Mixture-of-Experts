import torch
from sklearn.metrics import f1_score, classification_report
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
from Config import Config
import torch.nn.functional as F

def evaluate(model, data_loader, criterion=None):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            token_type_ids = batch["token_type_ids"].to(Config.device)
            labels = batch["label"].to(Config.device)
            options = batch["options"]

            outputs = model(input_ids, attention_mask, token_type_ids, options)
            
            #loss = criterion(outputs, labels)  # Calculate loss using the original labels
            loss = criterion(outputs, F.one_hot(labels, num_classes=Config.num_choices).long())  # Calculate loss using one-hot encoded labels

            total_loss += loss.item()

            outputs = outputs.view(outputs.shape[0], -1)  # Reshape the outputs tensor to [batch size, -1]
            _, predicted = torch.max(outputs, dim=1)  # Get predicted class labels

            #_, predicted = torch.max(outputs, dim=1)  # Get predicted class labels
            #_, labels = torch.max(labels, 1)

            # print("Outputs", outputs.shape)
            # print("Labels",labels.shape)
            # print("predictions:", predicted.shape)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    f1_score_macro = f1_score(all_labels, all_predictions, average='macro', labels=list(range(Config.num_choices)))
    class_report = classification_report(all_labels, all_predictions, labels=list(range(Config.num_choices)))

    return {
        "f1_score": f1_score_macro,
        "classification_report": class_report,
        "loss": total_loss / len(data_loader),  # Add average validation loss
        "labels": all_labels,  # Add the original labels
        "predictions": all_predictions,  # Add the predictions
    }
