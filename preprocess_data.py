from load_medqa_data import load_med_qa_data
from Config import Config

# Load data and create answer2id dictionary
def create_answer2id(train_data):
    answer_set = set(item['answer_idx'] for item in train_data)
    return {answer: i for i, answer in enumerate(answer_set)}

def preprocess_data(train_file, val_file, test_file):
    train_data, val_data, test_data = load_med_qa_data(train_file, val_file, test_file)
    Config.answer2id = create_answer2id(train_data)
    return train_data, val_data, test_data