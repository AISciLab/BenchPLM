import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             median_absolute_error, mean_absolute_percentage_error,
                             explained_variance_score)
from scipy.stats import pearsonr, spearmanr
from transformers import (
    Trainer, TrainingArguments, BertModel,
    TrainerCallback, EarlyStoppingCallback
)
import warnings
warnings.filterwarnings('ignore')

import tokenization
from tokenization import FullTokenizer

from data_utils import load_regression_data, get_max_lengths, normalize_regression_targets, denormalize_predictions
from transformers.trainer_utils import get_last_checkpoint

class AminobertPPIRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, protein_input_ids, protein_attention_mask, 
                peptide_input_ids, peptide_attention_mask, labels=None):
        protein_outputs = self.bert(input_ids=protein_input_ids, attention_mask=protein_attention_mask)
        protein_features = protein_outputs.pooler_output
        
        peptide_outputs = self.bert(input_ids=peptide_input_ids, attention_mask=peptide_attention_mask)
        peptide_features = peptide_outputs.pooler_output
        
        combined_features = torch.cat([protein_features, peptide_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        predictions = self.regressor(combined_features).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.view(-1), labels.view(-1))
            
        return {'loss': loss, 'logits': predictions} if loss is not None else {'logits': predictions}

class PPIDataset(Dataset):
    def __init__(self, protein_seqs, peptide_seqs, targets, tokenizer, max_protein_len=826, max_peptide_len=100):
        self.protein_seqs = protein_seqs
        self.peptide_seqs = peptide_seqs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_protein_len = max_protein_len
        self.max_peptide_len = max_peptide_len
    
    def __len__(self):
        return len(self.targets)
    
    def _process_sequence(self, seq, max_len):
        tokens = [tokenization.CLS_TOKEN] + self.tokenizer.tokenize(seq)
        
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        padding_len = max_len - len(tokens)
        padded_tokens = tokens + [tokenization.PAD_TOKEN] * padding_len
        
        input_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
        attention_mask = [1] * len(tokens) + [0] * padding_len
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, idx):
        protein_seq = str(self.protein_seqs[idx])
        peptide_seq = str(self.peptide_seqs[idx])
        
        protein_input_ids, protein_attention_mask = self._process_sequence(protein_seq, self.max_protein_len)
        peptide_input_ids, peptide_attention_mask = self._process_sequence(peptide_seq, self.max_peptide_len)
        
        return {
            'protein_input_ids': protein_input_ids,
            'protein_attention_mask': protein_attention_mask,
            'peptide_input_ids': peptide_input_ids,
            'peptide_attention_mask': peptide_attention_mask,
            'labels': torch.tensor(self.targets[idx], dtype=torch.float)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    medae = median_absolute_error(labels, predictions)
    evs = explained_variance_score(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'medae': medae,    
        'mape': mape,      
        'r2': r2,
        'evs': evs,        
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }

def train_and_evaluate(run_id, learning_rate=1e-5, num_epochs=100, batch_size=32):
    print(f"Starting Aminobert regression for run_{run_id}")
    
    model_path = './aminobert'
    output_dir = f'./output/regression/aminobert_regression_run_{run_id}'
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    model_save_dir = os.path.join(output_dir, 'model_checkpoint')

    data_path = "./dataset/PPI/regression"
    train_df, val_df, test_df = load_regression_data(data_path, run_id)
    
    max_lengths = get_max_lengths()['regression']
    
    train_targets = train_df['pKd'].values
    val_targets = val_df['pKd'].values
    test_targets = test_df['pKd'].values
    train_targets_norm, val_targets_norm, test_targets_norm, scaler = normalize_regression_targets(
        train_targets, val_targets, test_targets, method='standard'
    )
    
    print(f"Using custom FullTokenizer")
    tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')
    
    print(f"Loading model from local path: {model_path}")
    model = AminobertPPIRegressionModel(pretrained_model_name=model_path)
    
    train_dataset = PPIDataset(
        train_df['Protein_Sequence'].tolist(), train_df['Peptide_Sequence'].tolist(), train_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )
    val_dataset = PPIDataset(
        val_df['Protein_Sequence'].tolist(), val_df['Peptide_Sequence'].tolist(), val_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )
    test_dataset = PPIDataset(
        test_df['Protein_Sequence'].tolist(), test_df['Peptide_Sequence'].tolist(), test_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )
    
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        learning_rate=learning_rate,
        save_total_limit=1,
        remove_unused_columns=False,
        save_safetensors=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    print("Starting training...")
    
    last_checkpoint = get_last_checkpoint(model_save_dir)
    
    if last_checkpoint:
        print(f"Checkpoint found at {last_checkpoint}, resuming training.")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("Training finished.")
    
    test_results = trainer.evaluate(test_dataset)
    
    test_predictions = trainer.predict(test_dataset)
    test_pred_norm = test_predictions.predictions
    test_pred_original = denormalize_predictions(test_pred_norm, scaler)
    
    mse_original = mean_squared_error(test_targets, test_pred_original)
    rmse_original = np.sqrt(mse_original)
    mae_original = mean_absolute_error(test_targets, test_pred_original)
    r2_original = r2_score(test_targets, test_pred_original)
    pearson_original, _ = pearsonr(test_targets, test_pred_original)
    spearman_original, _ = spearmanr(test_targets, test_pred_original)
    medae_original = median_absolute_error(test_targets, test_pred_original)
    mape_original = mean_absolute_percentage_error(test_targets, test_pred_original)
    evs_original = explained_variance_score(test_targets, test_pred_original)
    
    print(f"AminoBERT Regression Run {run_id} Results (Original Scale):")
    print(f"Test MSE: {mse_original:.3f}")
    print(f"Test RMSE: {rmse_original:.3f}")
    print(f"Test MAE: {mae_original:.3f}")
    print(f"Test MedAE: {medae_original:.3f}")   
    print(f"Test MAPE: {mape_original:.3f}")    
    print(f"Test R²: {r2_original:.3f}")
    print(f"Test EVS: {evs_original:.3f}")      
    print(f"Test Pearson: {pearson_original:.3f}")
    print(f"Test Spearman: {spearman_original:.3f}")
    
    return {
        'mse': mse_original,
        'rmse': rmse_original,
        'mae': mae_original,
        'medae': medae_original,  
        'mape': mape_original,    
        'r2': r2_original,
        'evs': evs_original,      
        'pearson': pearson_original,
        'spearman': spearman_original
    }
    
results = train_and_evaluate(1)
