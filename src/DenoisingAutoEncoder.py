import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SpectrumDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


class TransEncoder(nn.Module):
    def __init__(
        self,
        input_dim=205,
        latent_dim=9,
        d_model=128,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # CLSトークン（1つ）を学習可能パラメータとして定義
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, D]

        # 1D入力 -> d_model次元へ投影
        self.input_proj = nn.Linear(1, d_model)  # [B, L, 1] → [B, L, D]

        # 位置埋め込み（CLS含めて +1）
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # x: [B, 205]
        B = x.size(0)
        x = x.unsqueeze(-1)  # [B, 205, 1]
        x = self.input_proj(x)  # [B, 205, D]

        # CLSトークンを各バッチに複製して先頭に追加
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 206, D]

        # 位置埋め込み加算
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer Encoder
        x = self.transformer_encoder(x)  # [B, 206, D]

        # 出力の先頭（CLSトークン）だけを取り出す
        cls_output = x[:, 0, :]  # [B, D]

        # 最終的な潜在表現に変換
        z = self.fc_out(cls_output)  # [B, latent_dim]
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256, p=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)


def augment(
    x,
    noise_range=(-0.05, 0.05),
    shift_range=(-0.2, 0.2),
    slope_range=(-0.2, 0.2),
    cutout_ratio=0.05,
    stretch_scale=1.2,
    n_apply=2
):
    B, L = x.shape
    device = x.device

    # ベクトル化された Aug 関数（x: [B_sub, L]）
    def aug_noise(x):
        scale = torch.empty(x.size(0), 1, device=x.device).uniform_(*noise_range)
        return x + torch.randn_like(x) * scale

    def aug_shift(x):
        shift = torch.empty(x.size(0), 1, device=x.device).uniform_(*shift_range)
        return x + shift

    def aug_slope(x):
        ramp = torch.linspace(-0.5, 0.5, steps=L, device=x.device).unsqueeze(0)
        ramp = ramp.expand(x.size(0), -1)
        slope = torch.empty(x.size(0), 1, device=x.device).uniform_(*slope_range)
        return x + slope * ramp

    def aug_cutout(x):
        cut_len = max(1, int(L * cutout_ratio))
        start_idxs = torch.randint(0, L - cut_len + 1, (x.size(0),), device=x.device)
        mask = torch.ones_like(x, dtype=torch.bool)
        idx_range = torch.arange(cut_len, device=x.device).unsqueeze(0)
        cut_idxs = (start_idxs.unsqueeze(1) + idx_range).clamp(0, L - 1)
        mask.scatter_(1, cut_idxs, False)
        return x.masked_fill(~mask, 0.0)

    def aug_stretch(x):
        x_ = x.unsqueeze(1)  # [B,1,L]
        new_len = int(L * stretch_scale)
        stretched = F.interpolate(x_, size=new_len, mode='linear', align_corners=False)
        return F.interpolate(stretched, size=L, mode='linear', align_corners=False).squeeze(1)

    aug_funcs = [aug_noise, aug_shift, aug_slope, aug_cutout, aug_stretch]
    n_aug = len(aug_funcs)

    # --- ランダムに2個ずつ選ぶ（[B, 2]） ---
    selected_aug_ids = torch.stack([
        torch.randperm(n_aug, device=device)[:n_apply] for _ in range(B)
    ])  # [B, n_apply]

    x_aug = x.clone()

    # --- 各Aug順に処理（逐次適用）---
    for step in range(n_apply):
        # 各サンプルに対してこのステップで選ばれたAugのインデックス
        step_ids = selected_aug_ids[:, step]  # [B]

        for aug_id in range(n_aug):
            mask = (step_ids == aug_id)
            if mask.any():
                x_aug[mask] = aug_funcs[aug_id](x_aug[mask])

    return x_aug


# Denoising Augmentation Transformer Autoencoder (DATA)

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = TransEncoder(latent_dim=self.latent_dim, input_dim=self.input_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim, output_dim=self.output_dim)

    def forward(self, x, augment_flag=True):
        if augment_flag:
            x_aug = augment(x)
        else:
            x_aug = x
        
        latent = self.encoder(x_aug)
        x_recon = self.decoder(latent)
        
        return x_recon, latent

def train(model, criterion, train_loader, val_loader, epochs, lr, patience=3, factor=0.1):
    
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)

    early_stop_patience = 10
    early_stop_counter = 0


    history = {"train_loss": [], "val_loss": []}
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        
        # --- Training ---
        
        model.train()
        train_loss = 0
        
        for x_batch in train_loader:
            x_batch = x_batch.to('cuda')
            optimizer.zero_grad()
            x_recon, latent = model(x_batch, augment_flag=True)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        history["train_loss"].append(train_loss)

        # --- Validation ---
        
        model.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to('cuda')
                x_recon, latent = model(x_batch, augment_flag=False)
                loss = criterion(x_recon, x_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        history["val_loss"].append(val_loss)
        
        print(f"[{epoch:02d}] Train MSE Loss: {train_loss:.5f} | Val MSE Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            early_stop_counter = 0

        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_state_dict, history


def get_results(model, data_loader):
    model = model.to('cuda')
    model.eval()
    latent_spaces = []
    recon = []
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to('cuda')
            x_recon, z = model(x_batch, augment_flag=False)

            latent_spaces.append(z.cpu().numpy())
            recon.append(x_recon.cpu().numpy())
    
    latent_spaces = np.concatenate(latent_spaces, axis=0)
    recon = np.concatenate(recon, axis=0)
    
    return latent_spaces, recon

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label='Train')
    plt.plot(epochs, history["val_loss"], label='Valid')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.xlim(1, max(epochs))
    plt.legend(loc='upper left')