# train.py
import argparse
import wandb
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import load_and_preprocess_data, tokenize_data
from utils import MyDataSet, compute_metrics, LABEL_MAP

def run_training(data_path, output_dir, epochs, batch_size, lr):
    # Initialize Weights & Biases tracking
    wandb.init(project="huggingface-mlops", name="distilbert-run-1")

    print("Loading and preparing text splits...")
    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(
        file_path=data_path, sample_size=5000
    )
    
    train_encodings, test_encodings = tokenize_data(train_texts, test_texts)

    # Convert dictionary distributions to torch dataset structure objects
    train_dataset = MyDataSet(train_encodings, train_labels)
    test_dataset = MyDataSet(test_encodings, test_labels)

    print("Loading sequence classification architecture model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(LABEL_MAP)
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training loop...")
    trainer.train()
    
    print(f"Saving final fine-tuned model checkpoint weights to: {output_dir}")
    trainer.save_model(output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT Text Classifier Training Pipeline")
    parser.add_argument('--data_path', type=str, default='books_genres.csv', help='Path to target data CSV')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save metrics/model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch sizing allocation')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial sequence learning rate')
    
    args = parser.parse_args()
    run_training(args.data_path, args.output_dir, args.epochs, args.batch_size, args.lr)