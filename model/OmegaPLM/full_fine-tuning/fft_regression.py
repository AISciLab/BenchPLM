import os
import sys
os.environ['INFRA_PROVIDER'] = 'omegafold'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             median_absolute_error, mean_absolute_percentage_error,
                             explained_variance_score)
from scipy.stats import pearsonr, spearmanr
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from omegafold import OmegaFold, config as omega_config, pipeline
from omegafold.utils import residue_constants as rc
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_regression_data, get_max_lengths, prepare_sequences_for_model, normalize_regression_targets, denormalize_predictions
from transformers.trainer_utils import get_last_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OmegaForSequenceRegression(torch.nn.Module):
    def __init__(self, model_idx=1):
        super().__init__()
        self.cfg = omega_config.make_config(model_idx)
        
        self.omega_plm = OmegaFold(self.cfg)
        
        weights_url = "https://helixon.s3.amazonaws.com/release1.pt"
        weights_file = "~/.cache/omegafold_ckpt/model.pt"
        state_dict = pipeline._load_weights(weights_url, weights_file)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.omega_plm.load_state_dict(state_dict, strict=False)
        
        self.regressor = nn.Sequential(
            nn.Linear(1280 * 2, 512),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.fwd_cfg = type('', (), {})() 
        self.fwd_cfg.subbatch_size = None

    def forward(self, protein_input_ids, protein_mask, peptide_input_ids, peptide_mask, labels=None):

        protein_repr, _ = self.omega_plm.omega_plm(protein_input_ids, protein_mask, fwd_cfg=self.fwd_cfg)
        protein_pooled = protein_repr.mean(dim=1)  
        
        peptide_repr, _ = self.omega_plm.omega_plm(peptide_input_ids, peptide_mask, fwd_cfg=self.fwd_cfg)
        peptide_pooled = peptide_repr.mean(dim=1)
        
        combined_features = torch.cat([protein_pooled, peptide_pooled], dim=1)
        
        predictions = self.regressor(combined_features).squeeze()
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(predictions, labels)
            
        return {'loss': loss, 'logits': predictions} if loss is not None else {'logits': predictions}

class OmegaCollator:
    def __init__(self, pad_token_id=21):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        protein_input_ids = [torch.tensor(item['protein_input_ids']) for item in batch]
        peptide_input_ids = [torch.tensor(item['peptide_input_ids']) for item in batch]
        
        protein_masks = [(ids != self.pad_token_id).float() for ids in protein_input_ids]
        peptide_masks = [(ids != self.pad_token_id).float() for ids in peptide_input_ids]
        
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        
        protein_input_ids = pad_sequence(protein_input_ids, batch_first=True, padding_value=self.pad_token_id)
        peptide_input_ids = pad_sequence(peptide_input_ids, batch_first=True, padding_value=self.pad_token_id)
        protein_masks = pad_sequence(protein_masks, batch_first=True, padding_value=0.0)
        peptide_masks = pad_sequence(peptide_masks, batch_first=True, padding_value=0.0)
        
        return {
            'protein_input_ids': protein_input_ids,  
            'protein_mask': protein_masks,
            'peptide_input_ids': peptide_input_ids,
            'peptide_mask': peptide_masks,
            'labels': labels        
        }

def tokenize_sequence(sequence):
    return [rc.restypes_with_x.index(aa) if aa in rc.restypes_with_x else 21 for aa in sequence]

class PPIDataset(Dataset):
    def __init__(self, protein_seqs, peptide_seqs, targets):
        self.protein_seqs = protein_seqs
        self.peptide_seqs = peptide_seqs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        protein_seq = str(self.protein_seqs[idx])
        peptide_seq = str(self.peptide_seqs[idx])
        
        protein_input_ids = tokenize_sequence(protein_seq)
        peptide_input_ids = tokenize_sequence(peptide_seq)
        
        return {
            'protein_input_ids': protein_input_ids,
            'peptide_input_ids': peptide_input_ids,
            'label': self.targets[idx]
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

def train_and_evaluate(run_id, learning_rate=5e-4, num_epochs=100, batch_size=6):
    print(f"Starting OmegaPLM regression for run_{run_id}")
    
    data_path = "./dataset/PPI/regression"
    output_dir = f'./output/regression/run_{run_id}'
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    model_save_dir = os.path.join(output_dir, 'model_checkpoint')
    train_df, val_df, test_df = load_regression_data(data_path, run_id)
    
    train_protein_seqs = train_df['Protein_Sequence'].tolist()
    train_peptide_seqs = train_df['Peptide_Sequence'].tolist()
    val_protein_seqs = val_df['Protein_Sequence'].tolist()
    val_peptide_seqs = val_df['Peptide_Sequence'].tolist()
    test_protein_seqs = test_df['Protein_Sequence'].tolist()
    test_peptide_seqs = test_df['Peptide_Sequence'].tolist()
    
    train_targets = train_df['pKd'].values
    val_targets = val_df['pKd'].values
    test_targets = test_df['pKd'].values
    
    train_targets_norm, val_targets_norm, test_targets_norm, scaler = normalize_regression_targets(
        train_targets, val_targets, test_targets, method='standard'
    )
    
    model = OmegaForSequenceRegression()
    
    train_dataset = PPIDataset(train_protein_seqs, train_peptide_seqs, train_targets_norm)
    val_dataset = PPIDataset(val_protein_seqs, val_peptide_seqs, val_targets_norm)
    test_dataset = PPIDataset(test_protein_seqs, test_peptide_seqs, test_targets_norm)
    
    data_collator = OmegaCollator()
    
    # 训练参数
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
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
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
    
    print(f"OmegaPLM Regression Run {run_id} Results (Original Scale):")
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
