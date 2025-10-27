import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             median_absolute_error, mean_absolute_percentage_error,
                             explained_variance_score)
from scipy.stats import pearsonr, spearmanr
from transformers import (
    Trainer, TrainingArguments, BertTokenizer,
    BertModel, TrainerCallback, EarlyStoppingCallback,TrainerState,TrainerControl
)
import warnings
warnings.filterwarnings('ignore')
import json

from data_utils import load_regression_data, get_max_lengths, prepare_sequences_for_model, normalize_regression_targets, denormalize_predictions
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model  

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
    
    def __getitem__(self, idx):
        protein_seq = str(self.protein_seqs[idx])
        peptide_seq = str(self.peptide_seqs[idx])
        
        protein_encoding = self.tokenizer(
            protein_seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_protein_len,
            return_tensors='pt'
        )
        
        peptide_encoding = self.tokenizer(
            peptide_seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_peptide_len,
            return_tensors='pt'
        )
        
        return {
            'protein_input_ids': protein_encoding['input_ids'].flatten(),
            'protein_attention_mask': protein_encoding['attention_mask'].flatten(),
            'peptide_input_ids': peptide_encoding['input_ids'].flatten(),
            'peptide_attention_mask': peptide_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.targets[idx], dtype=torch.float)
        }

class ProtBERTPPIRegressionModel(nn.Module):
    def __init__(self, model_name='Rostlab/prot_bert'):
        super().__init__()
        base_model = BertModel.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=8,                         
            lora_alpha=16,                
            target_modules=["query", "key", "value", "dense"],  
            lora_dropout=0.05,             
            bias="none",                  
        )
        
        self.protbert = get_peft_model(base_model, lora_config)
        self.protbert.print_trainable_parameters() 
        
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Sequential(
            nn.Linear(self.protbert.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, protein_input_ids, protein_attention_mask, 
                peptide_input_ids, peptide_attention_mask, labels=None):

        protein_outputs = self.protbert(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        )
        protein_features = protein_outputs.pooler_output

        peptide_outputs = self.protbert(
            input_ids=peptide_input_ids,
            attention_mask=peptide_attention_mask
        )
        peptide_features = peptide_outputs.pooler_output

        combined_features = torch.cat([protein_features, peptide_features], dim=1)
        combined_features = self.dropout(combined_features)

        predictions = self.regressor(combined_features).squeeze()
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)
        
        return {'loss': loss, 'logits': predictions} if loss is not None else {'logits': predictions}

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

def train_and_evaluate(run_id, model_name='model/prot_bert', learning_rate=2e-5, num_epochs=100, batch_size=8):
    
    print(f"Starting ProtBERT regression for run_{run_id}")

    data_path = "./dataset/PPI/regression"
    output_dir = f'./output/regression_lora/run_{run_id}'
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    model_save_dir = os.path.join(output_dir, 'model_checkpoint')
    
    train_df, val_df, test_df = load_regression_data(data_path, run_id)

    max_lengths = get_max_lengths()['regression']

    train_protein_seqs, train_peptide_seqs = prepare_sequences_for_model(
        train_df['Protein_Sequence'].tolist(), 
        train_df['Peptide_Sequence'].tolist(), 
        model_type='protbert'
    )
    val_protein_seqs, val_peptide_seqs = prepare_sequences_for_model(
        val_df['Protein_Sequence'].tolist(), 
        val_df['Peptide_Sequence'].tolist(), 
        model_type='protbert'
    )
    test_protein_seqs, test_peptide_seqs = prepare_sequences_for_model(
        test_df['Protein_Sequence'].tolist(), 
        test_df['Peptide_Sequence'].tolist(), 
        model_type='protbert'
    )

    train_targets = train_df['pKd'].values
    val_targets = val_df['pKd'].values
    test_targets = test_df['pKd'].values
    
    train_targets_norm, val_targets_norm, test_targets_norm, scaler = normalize_regression_targets(
        train_targets, val_targets, test_targets, method='standard'
    )

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = ProtBERTPPIRegressionModel(model_name)

    train_dataset = PPIDataset(
        train_protein_seqs, train_peptide_seqs, train_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )
    val_dataset = PPIDataset(
        val_protein_seqs, val_peptide_seqs, val_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )
    test_dataset = PPIDataset(
        test_protein_seqs, test_peptide_seqs, test_targets_norm,
        tokenizer, max_lengths['protein_max_len'], max_lengths['peptide_max_len']
    )

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False, 
        learning_rate=learning_rate,
        save_total_limit=1,
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_pin_memory=True,
        fp16=True,
        dataloader_num_workers=12
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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

    print(f"ProtBERT Regression Run {run_id} Results (Original Scale):")
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
