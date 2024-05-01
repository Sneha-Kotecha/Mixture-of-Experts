import json

def load_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_med_qa_data(train_file_path, dev_file_path, test_file_path):
    train_data = load_json_file(train_file_path)
    dev_data = load_json_file(dev_file_path)
    test_data = load_json_file(test_file_path)
    return train_data, dev_data, test_data


train_data, dev_data, test_data = load_med_qa_data('/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/train.jsonl',
                                                       '/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/dev.jsonl',
                                                       '/data/catz0136/Thesis_MoE/MedQA/data_clean/questions/US/test.jsonl')


# print("Train data structure:")
# for entry in train_data[:5]: # Print the structure of the first 5 entries
#     print(f"Question: {entry['question']}")
#     print(f"Options: {entry['options']}")
#     print(f"Answer: {entry['answer']}")
#     print()