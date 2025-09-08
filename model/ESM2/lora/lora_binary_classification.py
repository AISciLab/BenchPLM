import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, average_precision_score
)
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer,
    AutoModelForSequenceClassification, TrainerCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import warnings

warnings.filterwarnings('ignore')


# --- 1. Core Training Components ---

class EarlyStoppingCallback(TrainerCallback):

    def __init__(self, patience=5):
        self.patience = patience
        self.best_auc = 0
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_auc = metrics.get('eval_auc', 0)
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
            control.should_training_stop = True
        return control


def compute_metrics(p):

    preds = np.argmax(p.predictions, axis=1)
    probs = torch.nn.functional.softmax(
        torch.tensor(p.predictions), dim=-1).numpy()
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auc': roc_auc_score(labels, probs[:, 1]),
        'aupr': average_precision_score(labels, probs[:, 1]),
        'recall': recall_score(labels, preds),
        'precision': precision_score(labels, preds)
    }


# --- 2. Data Loaders ---

def load_toxteller_data(train_path, test_path):
    """Loads and prepares the ToxTeller dataset from FASTA files."""

    def read_fasta(file_path):
        sequences, labels = [], []
        seq = ""
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    if seq:
                        sequences.append(seq)
                    label = 1 if "pos" in line.lower() else 0
                    labels.append(label)
                    seq = ""
                else:
                    seq += line.strip()
            if seq:
                sequences.append(seq)
        return sequences, labels

    train_seqs, train_labels = read_fasta(train_path)
    test_seqs, test_labels = read_fasta(test_path)

    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels})
    return train_data, test_data


def load_hemopi_data(train_pos_path, train_neg_path, test_pos_path, test_neg_path):
    """Loads and prepares the HemoPI dataset from CSV files."""

    def read_csv_files(pos_path, neg_path):
        pos_data = pd.read_csv(pos_path, header=None)
        neg_data = pd.read_csv(neg_path, header=None)
        pos_sequences = pos_data[~pos_data[0].str.startswith('>')][0].str.upper().tolist()
        neg_sequences = neg_data[~neg_data[0].str.startswith('>')][0].str.upper().tolist()
        sequences = pos_sequences + neg_sequences
        labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)
        return sequences, labels

    train_seqs, train_labels = read_csv_files(train_pos_path, train_neg_path)
    test_seqs, test_labels = read_csv_files(test_pos_path, test_neg_path)

    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels})
    return train_data, test_data


def load_toxinpred_data(train_pos_path, train_neg_path, test_pos_path, test_neg_path):
    """Loads and prepares the ToxinPred 3.0 dataset from CSV files."""

    def read_csv_files(pos_path, neg_path):
        pos_data = pd.read_csv(pos_path, header=None)
        neg_data = pd.read_csv(neg_path, header=None)
        pos_data['label'] = 1
        neg_data['label'] = 0
        data = pd.concat([pos_data, neg_data], ignore_index=True)
        data[0] = data[0].apply(lambda x: ''.join([c.upper() if c in 'acdefghiklmnpqrstvwy' else c for c in x]))
        return data[0].tolist(), data['label'].tolist()

    train_seqs, train_labels = read_csv_files(train_pos_path, train_neg_path)
    test_seqs, test_labels = read_csv_files(test_pos_path, test_neg_path)

    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels})
    return train_data, test_data


# --- 3. Main Training and Evaluation Pipeline ---
def run_benchmark(dataset_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, args):

    print(f"\n===== Running ESM-2 LoRA Benchmark for: {dataset_name} =====")
    print(f"Run {args.run_time}/{args.num_folds}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='./model')

    def encode_data(batch):
        return tokenizer(batch['sequence'], truncation=True, padding='max_length', max_length=128)

    train_df, val_df = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=42)
    train_dataset = Dataset.from_pandas(train_df).map(encode_data, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(encode_data, batched=True)
    test_dataset = Dataset.from_pandas(test_data).map(encode_data, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        cache_dir='./model'
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    print("Trainable parameters after applying LoRA:")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f'./output/Lora_ESM2/{dataset_name}/Run_results_{args.run_time}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='auc',
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=f"./logs/Lora_ESM2/{dataset_name}/run_{args.run_time}",
        logging_steps=100,
        bf16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=args.patience)]
    )

    print("--- Starting ESM-2 LoRA Fine-Tuning ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the best model
    checkpoint_dir = f"./checkpoint/Lora_ESM2/{dataset_name}/ESM2_{args.run_time}"
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Best model and tokenizer saved to {checkpoint_dir}")

    print("--- Evaluating on Test Set ---")
    eval_results = trainer.evaluate(test_dataset)

    print("\nFinal Metrics on Test Set:")
    for metric in ['accuracy', 'f1', 'auc', 'aupr', 'recall', 'precision']:
        print(f"{metric}: {eval_results[f'eval_{metric}']:.3f}")

    print(f"===== ESM-2 LoRA Benchmark for {dataset_name} Finished =====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-2 3B LoRA Fine-Tuning Benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["ToxTeller", "ToxinPred", "HemoPI"],
                        help="Name of the dataset to run benchmark on.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t36_3B_UR50D",
                        help="Pre-trained ESM-2 model name.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER DEVICE for training and evaluation.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--num_folds", type=int, default=10, help="Total number of runs (for logging purposes).")
    parser.add_argument("--run_time", type=int, default=1, help="The current run index (e.g., 1 for the first run).")

    args = parser.parse_args()

    DATASET_CONFIG = {
        "ToxTeller": {
            "loader": load_toxteller_data,
            "paths": (
                "dataset/ToxTeller/training_dataset.fasta",
                "dataset/ToxTeller/independent_dataset.fasta"
            )
        },
        "ToxinPred": {
            "loader": load_toxinpred_data,
            "paths": (
                "dataset/Toxinpred3.0/train/train_pos.csv",
                "dataset/Toxinpred3.0/train/train_neg.csv",
                "dataset/Toxinpred3.0/test/test_pos.csv",
                "dataset/Toxinpred3.0/test/test_neg.csv"
            )
        },
        "HemoPI": {
            "loader": load_hemopi_data,
            "paths": (
                "dataset/HemoPI/train/train_pos.csv",
                "dataset/HemoPI/train/train_neg.csv",
                "dataset/HemoPI/val/val_pos.csv",
                "dataset/HemoPI/val/val_neg.csv"
            )
        }
    }

    if args.dataset in DATASET_CONFIG:
        config = DATASET_CONFIG[args.dataset]
        train_data, test_data = config["loader"](*config["paths"])
        run_benchmark(args.dataset, train_data, test_data, args)
    else:
        print(f"Error: Dataset '{args.dataset}' is not configured.")