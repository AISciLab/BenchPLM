import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def parse_affinity_to_pkd(affinity_str):

    if pd.isna(affinity_str) or affinity_str == '':
        return np.nan

    if 'pKd' in str(affinity_str) or 'pKi' in str(affinity_str):
        try:
            match = re.search(r'(\d+\.?\d*)', str(affinity_str))
            if match:
                return float(match.group(1))
        except:
            return np.nan
    
    try:
        affinity_str = str(affinity_str).strip()
        
        pattern = r'Kd?\s*=?\s*(\d+\.?\d*)\s*([a-zA-Z]+)'
        match = re.search(pattern, affinity_str, re.IGNORECASE)
        
        if not match:
            pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+)'
            match = re.search(pattern, affinity_str, re.IGNORECASE)
        
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            unit_conversion = {
                'fm': 1e-15,  # femtomolar
                'pm': 1e-12,  # picomolar
                'nm': 1e-9,   # nanomolar
                'um': 1e-6,   # micromolar
                'μm': 1e-6,   # micromolar
                'mm': 1e-3,   # millimolar
                'm': 1,       # molar
            }
            
            if unit in unit_conversion:
                kd_molar = value * unit_conversion[unit]
                pkd = -np.log10(kd_molar)
                return pkd
            else:
                print(f"Unknown unit: {unit} in {affinity_str}")
                return np.nan
        else:
            print(f"Could not parse: {affinity_str}")
            return np.nan
            
    except Exception as e:
        print(f"Error parsing {affinity_str}: {e}")
        return np.nan

def load_regression_data(data_path, run_id):

    train_path = f"{data_path}/run_{run_id}/train_set.csv"
    val_path = f"{data_path}/run_{run_id}/val_set.csv"
    test_path = f"{data_path}/run_{run_id}/test_set.csv"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df['pKd'] = train_df['Affinity'].apply(parse_affinity_to_pkd)
    val_df['pKd'] = val_df['Affinity'].apply(parse_affinity_to_pkd)
    test_df['pKd'] = test_df['Affinity'].apply(parse_affinity_to_pkd)
    
    train_df = train_df.dropna(subset=['pKd'])
    val_df = val_df.dropna(subset=['pKd'])
    test_df = test_df.dropna(subset=['pKd'])
    
    return train_df, val_df, test_df


def normalize_regression_targets(train_targets, val_targets, test_targets, method='standard'):

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    train_targets_scaled = scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
    val_targets_scaled = scaler.transform(val_targets.reshape(-1, 1)).flatten()
    test_targets_scaled = scaler.transform(test_targets.reshape(-1, 1)).flatten()
    
    return train_targets_scaled, val_targets_scaled, test_targets_scaled, scaler

def denormalize_predictions(predictions, scaler):

    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

def get_max_lengths():

    return {
        'classification': {
            'protein_max_len': 512,
            'peptide_max_len': 50
        },
        'regression': {
            'protein_max_len': 826,
            'peptide_max_len': 100
        }
    }

def clean_sequence(sequence):
    """
    Cleans protein sequences by replacing non-standard amino acid codes.
    This version replaces any character that is not an uppercase letter (A-Z) with 'X'.
    """
    # Ensure the input is a string to prevent errors
    if not isinstance(sequence, str):
        return ""

    # Use a regular expression to replace any non-alphabetic character with 'X'
    # This will handle '/', numbers, special symbols, etc.
    cleaned_seq = re.sub(r'[^A-Z]', 'X', sequence.upper())
    return cleaned_seq

def prepare_sequences_for_model(protein_seqs, peptide_seqs, model_type='bert'):

    if model_type in ['bert', 'protbert']:
        protein_seqs = [' '.join(list(clean_sequence(seq))) for seq in protein_seqs]
        peptide_seqs = [' '.join(list(clean_sequence(seq))) for seq in peptide_seqs]
    elif model_type in ['t5', 'prot_t5']:
        protein_seqs = [' '.join(list(clean_sequence(seq))) for seq in protein_seqs]
        peptide_seqs = [' '.join(list(clean_sequence(seq))) for seq in peptide_seqs]
    else:
        protein_seqs = [clean_sequence(seq) for seq in protein_seqs]
        peptide_seqs = [clean_sequence(seq) for seq in peptide_seqs]
    
    return protein_seqs, peptide_seqs
