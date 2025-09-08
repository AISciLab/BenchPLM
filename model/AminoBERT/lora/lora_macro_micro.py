import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer, TrainingArguments,
    BertForSequenceClassification, TrainerCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import tokenization
from tokenization import FullTokenizer
from evaluation import evaluate, scores
import warnings

warnings.filterwarnings('ignore')

PEPTIDE_TYPES = [
    'AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP',
    'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'
]


# --- 1. Core Model and Training Components ---

def process_seq(seqs, labels, tokenizer, max_length=128):
    """
    Custom function to tokenize and prepare sequences for AminoBERT.
    """
    tokenized = [[tokenization.CLS_TOKEN] + tokenizer.tokenize(s) for s in seqs]

    input_ids = []
    attention_mask = []

    for tokens in tokenized:
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        mask_len = len(tokens)
        tokens += [tokenization.PAD_TOKEN] * (max_length - len(tokens))

        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(ids)

        mask = [1] * mask_len + [0] * (max_length - mask_len)
        attention_mask.append(mask)

    return {
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_mask),
        'labels': np.array(labels, dtype=np.float32)
    }


class EarlyStoppingCallback(TrainerCallback):
    """
    Custom callback to stop training early based on macro accuracy.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.best_metric = 0
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get('eval_macro_accuracy', 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs with no improvement on macro_accuracy.")
            control.should_training_stop = True
        return control


def compute_metrics(p):
    """
    Combined metrics function to compute both macro and micro scores.
    """
    y_preds_logits = torch.tensor(p.predictions)
    y_preds_probs = torch.sigmoid(y_preds_logits).numpy()
    y_preds_binary = (y_preds_probs > 0.5).astype(int)
    y_true = p.label_ids

    metrics = {}

    # --- 1. Macro Metrics Calculation ---
    aiming, coverage, macro_accuracy, absolute_true, absolute_false = evaluate(y_preds_binary, y_true)
    metrics['macro_precision'] = aiming
    metrics['macro_coverage'] = coverage
    metrics['macro_accuracy'] = macro_accuracy
    metrics['macro_absolute_true'] = absolute_true
    metrics['macro_absolute_false'] = absolute_false

    # --- 2. Micro Metrics Calculation (per label) ---
    for i, label_name in enumerate(PEPTIDE_TYPES):
        label_true = y_true[:, i]
        label_pred_probs = y_preds_probs[:, i]

        recall, sn, sp, mcc, precision, f1, acc, auc, aupr, _, _, _, _ = scores(label_true, label_pred_probs)

        metrics[f'{label_name}_Recall'] = recall
        metrics[f'{label_name}_SN'] = sn
        metrics[f'{label_name}_SP'] = sp
        metrics[f'{label_name}_MCC'] = mcc
        metrics[f'{label_name}_Precision'] = precision
        metrics[f'{label_name}_F1'] = f1
        metrics[f'{label_name}_Acc'] = acc
        metrics[f'{label_name}_AUC'] = auc
        metrics[f'{label_name}_AUPR'] = aupr

    return metrics


def format_micro_results(eval_results):
    """
    Formats the micro-level evaluation results into a printable DataFrame.
    """
    metrics_order = ['Recall', 'SN', 'SP', 'MCC', 'Precision', 'F1', 'Acc', 'AUC', 'AUPR']

    formatted_data = []
    for label in PEPTIDE_TYPES:
        row = {'Label': label}
        for metric in metrics_order:
            key = f"eval_{label}_{metric}"
            if key in eval_results:
                row[metric] = round(eval_results[key], 4)
        formatted_data.append(row)

    return pd.DataFrame(formatted_data)[['Label'] + metrics_order]


# --- 2. Data Loader ---

def load_prmftp_data(train_path, test_path):
    """Loads and prepares the PrMFTP dataset."""

    def read_txt(file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    label_str = line[1:].strip()
                    labels.append([int(c) for c in label_str])
                else:
                    sequences.append(line.strip())
        return sequences, labels

    train_seqs, train_labels = read_txt(train_path)
    test_seqs, test_labels = read_txt(test_path)

    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels})
    return train_data, test_data


# --- 3. Main Training and Evaluation Pipeline ---

def run_benchmark(train_data: pd.DataFrame, test_data: pd.DataFrame, args):
    """
    The main function to run the AminoBERT LoRA benchmark for PrMFTP.
    """
    print(f"\n===== Running AminoBERT LoRA Benchmark for PrMFTP =====")
    print(f"Run {args.run_time}/{args.num_folds}")

    tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')

    # Split, process data, and create datasets
    train_df, val_df = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=42)

    train_processed = process_seq(train_df["sequence"].tolist(), train_df["label"].tolist(), tokenizer)
    val_processed = process_seq(val_df["sequence"].tolist(), val_df["label"].tolist(), tokenizer)
    test_processed = process_seq(test_data["sequence"].tolist(), test_data["label"].tolist(), tokenizer)

    train_dataset = Dataset.from_dict(train_processed)
    val_dataset = Dataset.from_dict(val_processed)
    test_dataset = Dataset.from_dict(test_processed)

    # Load base model for multi-label classification
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(PEPTIDE_TYPES),
        problem_type="multi_label_classification"
    )

    # PEFT LoRA Configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)
    print("Trainable parameters after applying LoRA:")
    model.print_trainable_parameters()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./output/Lora_PrMFTP/Run_{args.run_time}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=0.01,
        metric_for_best_model='eval_macro_accuracy',  # Use macro accuracy for model saving
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=f"./logs/Lora_PrMFTP/run_{args.run_time}",
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

    print("--- Starting AminoBERT LoRA Fine-Tuning for PrMFTP ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the best model
    checkpoint_dir = f"./checkpoint/Lora_PrMFTP/AminoBert_{args.run_time}"
    trainer.save_model(checkpoint_dir)
    print(f"Best model saved to {checkpoint_dir}")

    print("\n--- Evaluating on Test Set ---")
    eval_results = trainer.evaluate(test_dataset)

    print("\n--- Macro-level Metrics on Test Set ---")
    macro_metrics = ['macro_precision', 'macro_coverage', 'macro_accuracy', 'macro_absolute_true',
                     'macro_absolute_false']
    for metric in macro_metrics:
        print(f"{metric}: {eval_results[f'eval_{metric}']:.4f}")

    print("\n--- Micro-level Metrics on Test Set (Formatted) ---")
    results_df = format_micro_results(eval_results)
    print(results_df.to_string(index=False))

    csv_path = f"./results/Lora/PrMFTP_micro_results_run_{args.run_time}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\nMicro-level results saved to {csv_path}")

    print(f"===== PrMFTP Benchmark Finished =====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AminoBERT LoRA Benchmark for PrMFTP (Macro & Micro)")
    parser.add_argument("--model_name", type=str, default="./aminobert",
                        help="Path to the pre-trained AminoBERT model.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--num_folds", type=int, default=10, help="Total number of runs (for logging purposes).")
    parser.add_argument("--run_time", type=int, default=1, help="The current run index (e.g., 1 for the first run).")

    args = parser.parse_args()

    # Load data
    train_data, test_data = load_prmftp_data("dataset/PrMFTP/train.txt", "dataset/PrMFTP/test.txt")

    # Run the benchmark
    run_benchmark(train_data, test_data, args)