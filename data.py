# data.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from utils import LABEL_MAP

def load_and_preprocess_data(file_path, sample_size=None, test_size=0.2, random_state=42):
    """
    Loads raw CSV data, filters columns, samples text rows, maps textual genres 
    to integer IDs, and splits into train/test components.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Optional sampling to manage training times
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        
    # Standardize column mapping extract (assuming columns: 'text' and 'genre')
    texts = df['text'].astype(str).tolist()
    genres = df['genre'].tolist()
    
    # Encode target text string labels into integers
    labels = [LABEL_MAP[g] for g in genres]
    
    # Train-test split step
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    return train_texts, test_texts, train_labels, test_labels

def tokenize_data(train_texts, test_texts, model_name='distilbert-base-uncased'):
    """
    Applies huggingface fast tokenizer onto text structures.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    return train_encodings, test_encodings

if __name__ == "__main__":
    # Example execution pipeline saving processing outputs using pickle
    print("Executing standalone preprocessing checks...")
    # Replace with your local dataset path
    try:
        tr_txt, ts_txt, tr_lbl, ts_lbl = load_and_preprocess_data('books_genres.csv', sample_size=1000)
        tr_enc, ts_enc = tokenize_data(tr_txt, ts_txt)
        
        # Save split definitions for eval downstream consistency
        with open('processed_data.pkl', 'wb') as f:
            pickle.dump((ts_txt, ts_lbl), f)
        print("Data preparation complete and saved successfully.")
    except Exception as e:
        print(f"Skipping main processing: {e}. Provide an explicit CSV path to run.")