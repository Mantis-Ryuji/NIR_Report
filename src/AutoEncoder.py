import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

class StandardScaler:
    def __init__(self, ddof=1, eps=1e-8):
        self.ddof = ddof
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        X: np.ndarray of shape (N, D)
        """
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, ddof=self.ddof, keepdims=True)
        self.std_[self.std_ == 0] = self.eps  # ゼロ除算回避
        return self

    def transform(self, X):
        """
        Apply standardization using fitted mean and std
        """
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X_scaled):
        """
        Reconstruct original data from standardized
        """
        return X_scaled * self.std_ + self.mean_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class SpectrumDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

# -------------------------
# Encoder
# -------------------------

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pooling=True):
        super().__init__()
        self.use_pooling = use_pooling

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.activate = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if use_pooling else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.activate(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)

        out += residual
        out = self.activate(out)

        if self.use_pooling:
            out = self.pool(out)

        return out


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.layer1 = nn.Sequential(
            convblock(1, 64, use_pooling=True),   # //2
            convblock(64, 64, use_pooling=False)
        )
        self.layer2 = nn.Sequential(
            convblock(64, 128, use_pooling=True),  # //2
            convblock(128, 128, use_pooling=False)
        )
        self.layer3 = nn.Sequential(
            convblock(128, 256, use_pooling=True),  # //2
            convblock(256, 256, use_pooling=False)
        )
        
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, latent_dim)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_pool(out)
        out = self.flatten(out)
        out = self.activate(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

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

# -------------------------
# Decoder
# -------------------------
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


class AE(nn.Module):
    def __init__(self, latent_dim, input_dim, output_dim, encoder_type: str = 'Conv'):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        encoder_map = {
            'Conv': ConvEncoder,
            'Trans': TransEncoder,
        }

        if encoder_type not in encoder_map:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be one of {list(encoder_map.keys())}")

        self.encoder = encoder_map[encoder_type](latent_dim=self.latent_dim, input_dim=self.input_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim, output_dim=self.output_dim)

    def forward(self, x):
        latent = self.encoder(x)
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
            x_recon, latent = model(x_batch)
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
                x_recon, latent = model(x_batch)
                loss = criterion(x_recon, x_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        history["val_loss"].append(val_loss)
        
        print(f"[{epoch:02d}] Train L1 Loss: {train_loss:.5f} | Val L1 Loss: {val_loss:.5f}")

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

def test(model, criterion, test_loader):
    model = model.to('cuda') 
    model.eval()
    
    test_loss = 0
    
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch.to('cuda')
            x_recon, latent = model(x_batch)
            loss = criterion(x_recon, x_batch)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test L1 Loss: {test_loss:.5f}')

def get_results(model, data_loader):
    model = model.to('cuda')
    model.eval()
    latent_spaces = []
    recon = []
    original = []
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to('cuda')
            x_recon, z = model(x_batch)

            latent_spaces.append(z.cpu().numpy())
            recon.append(x_recon.cpu().numpy())
            original.append(x_batch.cpu().numpy())
    
    latent_spaces = np.concatenate(latent_spaces, axis=0)
    recon = np.concatenate(recon, axis=0)
    original = np.concatenate(original, axis=0)
    
    return latent_spaces, recon, original

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label='Train')
    plt.plot(epochs, history["val_loss"], label='Valid')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.xlim(1, max(epochs))
    plt.legend(loc='upper left')