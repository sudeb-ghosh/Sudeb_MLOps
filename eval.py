# eval.py
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from data import load_and_preprocess_data
from utils import LABEL_MAP, INV_LABEL_MAP

def evaluate_model(model_dir, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inferential metrics evaluation pipeline using device target: {device}")

    # Load trained artifacts
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Load ground evaluation test components
    _, test_texts, _, test_labels = load_and_preprocess_data(file_path=data_path, sample_size=5000)

    predicted_labels = []
    
    print("Evaluating target validation texts predictions loop...")
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            predicted_labels.append(pred_id)

    # Convert encoded structural IDs back to actual string tags
    true_genres = [INV_LABEL_MAP[l] for l in test_labels]
    pred_genres = [INV_LABEL_MAP[p] for p in predicted_labels]

    # Structuring counts dictionary tracking metrics 
    genre_classifications_dict = defaultdict(int)
    for t, p in zip(true_genres, pred_genres):
        genre_classifications_dict[(t, p)] += 1

    dicts_to_plot = []
    for (true_g, pred_g), count in genre_classifications_dict.items():
        dicts_to_plot.append({'True Genre': true_g, 'Predicted Genre': pred_g, 'Number of Classifications': count})

    df_to_plot = pd.DataFrame(dicts_to_plot)
    df_wide = df_to_plot.pivot_table(index='True Genre', columns='Predicted Genre', values='Number of Classifications').fillna(0)

    # Output evaluation results to csv
    df_wide.to_csv('evaluation_matrix.csv')
    print("Matrix results saved under evaluation_matrix.csv")

    # Generate Confusion Matrix Heatmap Visualization
    plt.figure(figsize=(9, 7))
    sns.set(style='ticks', font_scale=1.2)
    sns.heatmap(df_wide, linewidths=1, cmap='Purples', annot=True, fmt='g')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap.png')
    print("Plot visualization exported successfully as 'confusion_matrix_heatmap.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation & Verification Execution Pipeline")
    parser.add_argument('--model_dir', type=str, default='./results', help='Path pointing towards your model checkpoint directory')
    parser.add_argument('--data_path', type=str, default='books_genres.csv', help='Original path pointing to data source')
    
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.data_path)print("Sudeb")