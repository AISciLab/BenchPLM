import os
import sys
os.environ['INFRA_PROVIDER'] = 'CARP'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             median_absolute_error, mean_absolute_percentage_error,
                             explained_variance_score)
from scipy.stats import pearsonr, spearmanr
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback,TrainerState,TrainerControl
from sequence_models.pretrained import load_model_and_alphabet
import warnings
warnings.filterwarnings('ignore')
import json
from data_utils import load_regression_data, get_max_lengths, prepare_sequences_for_model, normalize_regression_targets, denormalize_predictions
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PersistentEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.state_file = "early_stopping_state.json"

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        
        if args.resume_from_checkpoint:
            state_path = os.path.join(args.resume_from_checkpoint, self.state_file)
            if os.path.exists(state_path):
                print(f"Loading early stopping state from {state_path}")
                with open(state_path, "r") as f:
                    saved_state = json.load(f)
                self.early_stopping_patience_counter = saved_state.get("early_stopping_patience_counter", 0)
                print(f"Restored patience counter to: {self.early_stopping_patience_counter}")
            else:
                print("No early stopping state file found. Starting with a fresh counter.")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            state_path = os.path.join(checkpoint_folder, self.state_file)
        
            current_state = {"early_stopping_patience_counter": self.early_stopping_patience_counter}
            
            print(f"Saving early stopping state to {state_path}. Current patience: {self.early_stopping_patience_counter}")
            with open(state_path, "w") as f:
                json.dump(current_state, f)

class CARPFinetuneRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.carp_model, self.collater = load_model_and_alphabet('model/carp_640M.pt')
        
        lora_config = LoraConfig(
            r=8,                        
            lora_alpha=16,              
            target_modules=["sequence1.2.conv","sequence2.2.conv"],  
            lora_dropout=0.05,          
            bias="none",               
        )
        
        self.carp_model = get_peft_model(self.carp_model, lora_config)
        self.carp_model.print_trainable_parameters()  
        
        self.regressor = nn.Sequential(
            nn.Linear(1280 * 2, 512),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, protein_input_ids, peptide_input_ids, labels=None):

        protein_rep = self.carp_model(protein_input_ids, repr_layers=[32], logits=True)
        protein_embeddings = protein_rep['representations'][32].mean(1)
        
        peptide_rep = self.carp_model(peptide_input_ids, repr_layers=[32], logits=True)
        peptide_embeddings = peptide_rep['representations'][32].mean(1)
        
        combined_features = torch.cat([protein_embeddings, peptide_embeddings], dim=1)
        
        predictions = self.regressor(combined_features).squeeze()

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss = nn.MSELoss()(predictions, labels)

        return {'loss': loss, 'logits': predictions} if loss is not None else {'logits': predictions}

class CARPCollator:
    def __init__(self, collater):
        self.collater = collater

    def __call__(self, batch):
        protein_sequences = [item['protein_sequence'] for item in batch]
        peptide_sequences = [item['peptide_sequence'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        
        protein_seqs = [[seq] for seq in protein_sequences]
        peptide_seqs = [[seq] for seq in peptide_sequences]
        
        protein_input_ids = self.collater(protein_seqs)[0]
        peptide_input_ids = self.collater(peptide_seqs)[0]
        
        return {
            'protein_input_ids': protein_input_ids, 
            'peptide_input_ids': peptide_input_ids,
            'labels': labels
        }

class PPIDataset(Dataset):
    def __init__(self, protein_seqs, peptide_seqs, targets):
        self.protein_seqs = protein_seqs
        self.peptide_seqs = peptide_seqs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'protein_sequence': self.protein_seqs[idx], 
            'peptide_sequence': self.peptide_seqs[idx],
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

def train_and_evaluate(run_id, learning_rate=2e-5, num_epochs=100, batch_size=8):
    print(f"Starting CARP regression for run_{run_id}")
    
    data_path = "./dataset/PPI/regression"
    output_dir = f'./output/regression_lora/run_{run_id}'
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    model_save_dir = os.path.join(output_dir, 'model_checkpoint')
    train_df, val_df, test_df = load_regression_data(data_path, run_id)
    
    train_protein_seqs, train_peptide_seqs = prepare_sequences_for_model(
        train_df['Protein_Sequence'].tolist(),
        train_df['Peptide_Sequence'].tolist(),
        model_type='other'
    )
    val_protein_seqs, val_peptide_seqs = prepare_sequences_for_model(
        val_df['Protein_Sequence'].tolist(),
        val_df['Peptide_Sequence'].tolist(),
        model_type='other'
    )
    test_protein_seqs, test_peptide_seqs = prepare_sequences_for_model(
        test_df['Protein_Sequence'].tolist(),
        test_df['Peptide_Sequence'].tolist(),
        model_type='other'
    )
    
    train_targets = train_df['pKd'].values
    val_targets = val_df['pKd'].values
    test_targets = test_df['pKd'].values
    
    train_targets_norm, val_targets_norm, test_targets_norm, scaler = normalize_regression_targets(
        train_targets, val_targets, test_targets, method='standard'
    )
    
    model = CARPFinetuneRegressionModel()
    
    train_dataset = PPIDataset(train_protein_seqs, train_peptide_seqs, train_targets_norm)
    val_dataset = PPIDataset(val_protein_seqs, val_peptide_seqs, val_targets_norm)
    test_dataset = PPIDataset(test_protein_seqs, test_peptide_seqs, test_targets_norm)
    
    data_collator = CARPCollator(model.collater)
    
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
        dataloader_pin_memory=False,
        save_safetensors=False,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PersistentEarlyStoppingCallback(early_stopping_patience=5)]
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
    
    print(f"CARP Regression Run {run_id} Results (Original Scale):")
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
