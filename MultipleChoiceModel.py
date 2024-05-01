import torch
import torch.nn as nn
from transformers import BertConfig
import numpy as np
from Config import Config
from Bert_Class import BERT
from MoE_Class import MoE

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MultipleChoiceModel(nn.Module):
    def __init__(self, pretrained_model_name, num_choices, num_experts=5, tokenizer=None):
        super(MultipleChoiceModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model_name)
        self.bert = BERT(config)
        self.moe = MoE(num_experts, config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_choices)
        self.tokenizer = tokenizer  # Add tokenizer to the model
        self.max_length = Config.max_length  # Add max_length to the model

    def forward(self, input_ids, attention_mask, token_type_ids, options):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = torch.mean(sequence_output, dim=1)

        options_embeddings = []
        for option_text in options.values():
            option_encoding = self.tokenizer(option_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            option_input_ids = option_encoding["input_ids"].to(Config.device)
            option_embedding = torch.mean(self.bert.bert(option_input_ids)[0], dim=1)
            options_embeddings.append(option_embedding)

        options_embeddings = torch.stack(options_embeddings, dim=1).unsqueeze(1)
        pooled_output = pooled_output.unsqueeze(1).unsqueeze(1).repeat(1, 1, options_embeddings.size(2), 1)
        combined_output = self.moe(torch.cat([pooled_output, options_embeddings], dim=2).squeeze(1))
        logits = self.classifier(combined_output)
        return logits
    

