import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, TrainerCallback
from sequence_models.pretrained import load_model_and_alphabet
from evaluation import evaluate, scores
import warnings

os.environ['INFRA_PROVIDER'] = 'CARP'
warnings.filterwarnings('ignore')

PEPTIDE_TYPES = [
    'AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP',
    'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Core Model and Training Components ---

class CARPFinetuneModel(nn.Module):
    """
    The core CARP model with a classification head for full fine-tuning.
    """

    def __init__(self):
        super().__init__()
        self.carp_model, self.collater = load_model_and_alphabet('model/carp_640M.pt')
        self.classifier = nn.Linear(1280, len(PEPTIDE_TYPES))

    def forward(self, input_ids, labels=None):
        rep = self.carp_model(input_ids, repr_layers=[32], logits=True)
        embeddings = rep['representations'][32].mean(1)
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        return (loss, logits) if loss is not None else logits


class CARPCollator:
    """
    Custom data collator for the CARP model.
    """

    def __init__(self, collater):
        self.collater = collater

    def __call__(self, batch):
        sequences = [item['sequence'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        seqs = [[seq] for seq in sequences]
        input_ids = self.collater(seqs)[0]
        return {'input_ids': input_ids, 'labels': labels}


class ProteinDataset(Dataset):
    """
    Custom PyTorch Dataset for protein sequences.
    """

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'sequence': self.sequences.iloc[idx], 'label': self.labels.iloc[idx]}


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
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')

    def read_txt(file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    label_str = line[1:].strip()
                    labels.append([int(c) for c in label_str])
                else:
                    seq = line.strip().upper()
                    filtered_seq = ''.join([c for c in seq if c in valid_chars])
                    sequences.append(filtered_seq)
        return sequences, np.array(labels, dtype=np.float32)

    train_seqs, train_labels = read_txt(train_path)
    test_seqs, test_labels = read_txt(test_path)

    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels.tolist()})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels.tolist()})
    return train_data, test_data


# --- 3. Main Training and Evaluation Pipeline ---

def run_benchmark(train_data: pd.DataFrame, test_data: pd.DataFrame, args):
    """
    The main function to run the CARP FFT benchmark for PrMFTP.
    """
    print(f"\n===== Running CARP FFT Benchmark for PrMFTP =====")
    print(f"Run {args.run_time}/{args.num_folds}")

    train_df, val_df = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=42)
    train_dataset = ProteinDataset(train_df["sequence"], train_df["label"])
    val_dataset = ProteinDataset(val_df["sequence"], val_df["label"])
    test_dataset = ProteinDataset(test_data["sequence"], test_data["label"])

    model = CARPFinetuneModel()
    collator = CARPCollator(model.collater)

    training_args = TrainingArguments(
        output_dir=f'./output/FFT_CARP_PrMFTP/Run_{args.run_time}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=0.01,
        metric_for_best_model='eval_macro_accuracy',
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=f"./logs/FFT_CARP_PrMFTP/run_{args.run_time}",
        logging_steps=100,
        bf16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=args.patience)]
    )

    print("--- Starting CARP Full Fine-Tuning for PrMFTP ---")
    trainer.train()
    print("--- Training Finished ---")

    checkpoint_dir = f"./checkpoint/FFT_CARP_PrMFTP/CARP_{args.run_time}"
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

    csv_path = f"./results/FFT/CARP_PrMFTP_micro_results_run_{args.run_time}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\nMicro-level results saved to {csv_path}")

    print(f"===== PrMFTP Benchmark Finished =====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARP FFT Benchmark for PrMFTP (Macro & Micro)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--num_folds", type=int, default=10, help="Total number of runs (for logging purposes).")
    parser.add_argument("--run_time", type=int, default=1, help="The current run index (e.g., 1 for the first run).")

    args = parser.parse_args()

    train_data, test_data = load_prmftp_data("dataset/PrMFTP/train.txt", "dataset/PrMFTP/test.txt")

    run_benchmark(train_data, test_data, args)