import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, TrainerCallback
from omegafold import OmegaFold, config as omega_config, pipeline
from omegafold.utils import residue_constants as rc
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from evaluation import evaluate, scores
import warnings

os.environ['INFRA_PROVIDER'] = 'omegafold'
warnings.filterwarnings('ignore')

PEPTIDE_TYPES = [
    'AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP',
    'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Core Model and Training Components ---

class OmegaForSequenceClassification(torch.nn.Module):
    """
    Custom module for OmegaFold sequence classification with LoRA.
    """

    def __init__(self, model_idx=1):
        super().__init__()
        self.cfg = omega_config.make_config(model_idx)
        self.omega_plm = OmegaFold(self.cfg)

        # Load pre-trained weights
        weights_url = "https://helixon.s3.amazonaws.com/release1.pt"
        weights_file = "~/.cache/omegafold_ckpt/model.pt"
        state_dict = pipeline._load_weights(weights_url, weights_file)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.omega_plm.load_state_dict(state_dict, strict=False)

        # LoRA Configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "gva_proj.0", "out_proj", "plm_node_embedder", "plm_edge_embedder", "proj_edge_bias",
                "network.0", "network.2", "network.4", "out_product.input_proj", "node_final_proj",
                "init_proj", "transition.0", "transition.1", "transition.2", "input_projection.0", "input_projection.1",
                "resblock1.0", "resblock1.1", "resblock2.0", "resblock2.1", "unnormalized_angles",
                "q_scalar", "k_scalar", "v_scalar", "q_point", "k_point", "v_point", "bias_2d", "output_projection",
            ],
            lora_dropout=0.05,
            bias="none",
        )
        self.omega_plm = get_peft_model(self.omega_plm, lora_config)
        print("Trainable parameters after applying LoRA:")
        self.omega_plm.print_trainable_parameters()

        self.classifier = nn.Linear(self.cfg.model.node_dim, len(PEPTIDE_TYPES))

        self.fwd_cfg = type('', (), {})()
        self.fwd_cfg.subbatch_size = None

    def forward(self, input_ids, mask, labels=None):
        node_repr, _ = self.omega_plm.omega_plm(input_ids, mask, fwd_cfg=self.fwd_cfg)
        pooled = node_repr.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.to(device))

        return {'loss': loss, 'logits': logits}


class OmegaCollator:
    def __init__(self, pad_token_id=21):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        masks = [(ids != self.pad_token_id).float() for ids in input_ids]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        masks = pad_sequence(masks, batch_first=True, padding_value=0.0)

        return {'input_ids': input_ids, 'mask': masks, 'labels': labels}


def tokenize_sequence(sequence):
    return [rc.restypes_with_x.index(aa) if aa in rc.restypes_with_x else 21 for aa in sequence]


def encode_data(batch):
    return {
        'input_ids': [tokenize_sequence(seq) for seq in batch['sequence']],
        'label': batch['label']
    }


class EarlyStoppingCallback(TrainerCallback):
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
    y_preds_logits = torch.tensor(p.predictions)
    y_preds_probs = torch.sigmoid(y_preds_logits).numpy()
    y_preds_binary = (y_preds_probs > 0.5).astype(int)
    y_true = p.label_ids
    metrics = {}

    aiming, coverage, macro_accuracy, absolute_true, absolute_false = evaluate(y_preds_binary, y_true)
    metrics['macro_precision'] = aiming
    metrics['macro_coverage'] = coverage
    metrics['macro_accuracy'] = macro_accuracy
    metrics['macro_absolute_true'] = absolute_true
    metrics['macro_absolute_false'] = absolute_false

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
        return sequences, np.array(labels, dtype=np.float32)

    train_seqs, train_labels = read_txt(train_path)
    test_seqs, test_labels = read_txt(test_path)
    train_data = pd.DataFrame({"sequence": train_seqs, "label": train_labels.tolist()})
    test_data = pd.DataFrame({"sequence": test_seqs, "label": test_labels.tolist()})
    return train_data, test_data


# --- 3. Main Training and Evaluation Pipeline ---

def run_benchmark(train_data: pd.DataFrame, test_data: pd.DataFrame, args):
    print(f"\n===== Running OmegaFold LoRA Benchmark for PrMFTP =====")
    print(f"Run {args.run_time}/{args.num_folds}")

    train_df, val_df = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=42)
    train_dataset = Dataset.from_pandas(train_df).map(encode_data, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(encode_data, batched=True)
    test_dataset = Dataset.from_pandas(test_data).map(encode_data, batched=True)

    model = OmegaForSequenceClassification().to(device)

    training_args = TrainingArguments(
        output_dir=f'./output/Lora_OmegaFold_PrMFTP/Run_{args.run_time}',
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
        logging_dir=f"./logs/Lora_OmegaFold_PrMFTP/run_{args.run_time}",
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
        data_collator=OmegaCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=args.patience)]
    )

    print("--- Starting OmegaFold LoRA Fine-Tuning for PrMFTP ---")
    trainer.train()
    print("--- Training Finished ---")

    checkpoint_dir = f"./checkpoint/Lora_OmegaFold_PrMFTP/OmegaPLM_{args.run_time}"
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

    csv_path = f"./results/Lora/OmegaFold_PrMFTP_micro_results_run_{args.run_time}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\nMicro-level results saved to {csv_path}")

    print(f"===== PrMFTP Benchmark Finished =====\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmegaFold LoRA Benchmark for PrMFTP (Macro & Micro)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--num_folds", type=int, default=10, help="Total number of runs (for logging purposes).")
    parser.add_argument("--run_time", type=int, default=1, help="The current run index (e.g., 1 for the first run).")

    args = parser.parse_args()

    train_data, test_data = load_prmftp_data("dataset/PrMFTP/train.txt", "dataset/PrMFTP/test.txt")

    run_benchmark(train_data, test_data, args)