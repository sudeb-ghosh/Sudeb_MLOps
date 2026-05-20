# utils.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Label mapping definitions derived from the dataset
LABEL_MAP = {
    'fantasy': 0,
    'science_fiction': 1,
    'mystery_thriller_crime': 2,
    'history_biography': 3,
    'romance': 4,
    'young_adult': 5,
    'children': 6,
    'poetry': 7
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class MyDataSet(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading text encodings and corresponding labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """
    Computes classification evaluation metrics for the Hugging Face Trainer loop.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }