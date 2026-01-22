"""Quick test of tokenizer encode/decode"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')
from model.kronos import KronosTokenizer
from config import Config
import pickle

config = Config()
device = 'cuda'

# Load tokenizer
path = config.finetuned_tokenizer_path + '/tokenizer.pt'
checkpoint = torch.load(path, map_location=device, weights_only=False)
arch = checkpoint['config']['tokenizer_arch']

tokenizer = KronosTokenizer(
    arch['d_in'], arch['d_model'], arch['n_heads'], arch['ff_dim'],
    arch['n_enc_layers'], arch['n_dec_layers'],
    arch['ffn_dropout_p'], arch['attn_dropout_p'], arch['resid_dropout_p'],
    arch['s1_bits'], arch['s2_bits'],
    arch['beta'], arch['gamma0'], arch['gamma'], arch['zeta'], arch['group_size']
)
tokenizer.load_state_dict(checkpoint['model_state_dict'])
tokenizer.eval().to(device)

print('='*60)
print('TOKENIZER CODEBOOK UTILIZATION TEST')
print('='*60)
print(f'd_in: {arch["d_in"]}, s1_bits: {arch["s1_bits"]}, s2_bits: {arch["s2_bits"]}')
print(f'Max possible tokens: s1={2**arch["s1_bits"]}, s2={2**arch["s2_bits"]}')

# Load validation data
val_path = config.dataset_path + '/val_data.pkl'
with open(val_path, 'rb') as f:
    val_data = pickle.load(f)

print(f'\nProcessing {len(val_data)} symbols...')

all_tokens_s1 = []
all_tokens_s2 = []
all_mse = []

for symbol in list(val_data.keys())[:20]:  # First 20 symbols
    df = val_data[symbol]
    feature_list = config.feature_list
    available = [f for f in feature_list if f in df.columns]
    
    # Get multiple windows
    for start in range(0, min(500, len(df) - 50), 50):
        x = df[available].iloc[start:start+50].values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        
        # Normalize
        median = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1e-8
        x_norm = (x - median) / iqr
        x_norm = np.clip(x_norm, -3, 3)
        
        if x_norm.shape[1] < arch['d_in']:
            padded = np.zeros((x_norm.shape[0], arch['d_in']), dtype=np.float32)
            padded[:, :x_norm.shape[1]] = x_norm
            x_norm = padded
        
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
        
        with torch.no_grad():
            tokens = tokenizer.encode(x_tensor, half=True)
            all_tokens_s1.extend(tokens[0].flatten().cpu().tolist())
            all_tokens_s2.extend(tokens[1].flatten().cpu().tolist())
            
            decoded = tokenizer.decode(tokens, half=True)
            mse = ((x_tensor - decoded) ** 2).mean().item()
            all_mse.append(mse)

print(f'\nTotal tokens processed: {len(all_tokens_s1)}')
print(f'\nToken S1 (first half):')
print(f'  Unique values: {len(set(all_tokens_s1))} / {2**arch["s1_bits"]} possible')
print(f'  Most common: {sorted([(all_tokens_s1.count(t), t) for t in set(all_tokens_s1)], reverse=True)[:10]}')

print(f'\nToken S2 (second half):')
print(f'  Unique values: {len(set(all_tokens_s2))} / {2**arch["s2_bits"]} possible')
print(f'  Most common: {sorted([(all_tokens_s2.count(t), t) for t in set(all_tokens_s2)], reverse=True)[:10]}')

print(f'\nReconstruction MSE:')
print(f'  Mean: {np.mean(all_mse):.6f}')
print(f'  Std:  {np.std(all_mse):.6f}')
print(f'  Min:  {np.min(all_mse):.6f}')
print(f'  Max:  {np.max(all_mse):.6f}')
