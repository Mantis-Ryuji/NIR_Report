import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from kneed import KneeLocator

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

class EuclideanKMeans:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, device='cuda', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.random_state = random_state
        self.inertia_ = None

    def _euclidean_distance(self, A, B):
        # A: (N, D), B: (K, D) → 出力: (N, K)
        A_sq = (A ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        B_sq = (B ** 2).sum(dim=1).unsqueeze(0)   # (1, K)
        dist = A_sq + B_sq - 2 * (A @ B.T)
        return dist  # (N, K)

    def fit_predict(self, X):
        X = X.to(self.device)
        N, D = X.shape

        # 初期中心（ランダム選択）
        if self.random_state is not None:
            g = torch.Generator().manual_seed(self.random_state)
            indices = torch.randperm(N, generator=g)[:self.n_clusters]
        else:
            indices = torch.randperm(N)[:self.n_clusters]

        centroids = X[indices].clone()

        for _ in range(self.max_iter):
            dist = self._euclidean_distance(X, centroids)  # (N, K)
            labels = torch.argmin(dist, dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if members.shape[0] == 0:
                    new_centroids[k] = centroids[k]  # 空クラスタは据え置き
                else:
                    new_centroids[k] = members.mean(dim=0)

            shift = torch.norm(centroids - new_centroids).item()
            centroids = new_centroids
            if shift < self.tol:
                break

        # 最終的なラベルと inertia（＝総平方距離）
        final_dist = self._euclidean_distance(X, centroids)
        self.labels_ = torch.argmin(final_dist, dim=1)
        self.cluster_centers_ = centroids
        best_dist = final_dist[torch.arange(N), self.labels_]
        self.inertia_ = best_dist.sum().item()

        return self.labels_.cpu().numpy()


class CosineKMeans:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, device='cuda', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.inertia_ = None
        self.random_state = random_state

    def _cosine_similarity(self, A, B):
        A_norm = F.normalize(A, dim=1)
        B_norm = F.normalize(B, dim=1)
        return A_norm @ B_norm.T  # (N, K)

    def fit_predict(self, X):
        X = X.to(self.device)
        N, D = X.shape

        # ランダム初期化
        if self.random_state is not None:
            g = torch.Generator().manual_seed(self.random_state)
            indices = torch.randperm(N, generator=g)[:self.n_clusters]
        else:
            indices = torch.randperm(N)[:self.n_clusters]

        centroids = X[indices].clone()
        centroids = F.normalize(centroids, dim=1)

        for _ in range(self.max_iter):
            sim = self._cosine_similarity(X, centroids)  # (N, K)
            labels = torch.argmax(sim, dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if members.shape[0] == 0:
                    new_centroids[k] = centroids[k]
                else:
                    mean = members.mean(dim=0)
                    new_centroids[k] = F.normalize(mean, dim=0)
            
            center_sim = F.cosine_similarity(centroids, new_centroids, dim=1)
            min_sim = center_sim.min().item()
            if 1.0 - min_sim < self.tol:
                break

            centroids = new_centroids

        # 最終的なラベルと inertia
        final_sim = self._cosine_similarity(X, centroids)
        self.labels_ = torch.argmax(final_sim, dim=1)
        self.cluster_centers_ = centroids
        best_sim = final_sim[torch.arange(N), self.labels_]
        self.inertia_ = torch.sum(1.0 - best_sim).item()

        return self.labels_.cpu().numpy()


def elbow_method_with_knee(X, k_range=range(1, 51), random_state=42, type='cosine'):
    # クラスタリングモデル選択
    if type == 'cosine':
        ClusterModel = CosineKMeans
    elif type == 'euclidean':
        ClusterModel = EuclideanKMeans
    else:
        raise ValueError("type must be 'cosine' or 'euclidean'")

    inertias = []

    for k in k_range:
        model = ClusterModel(n_clusters=k, random_state=random_state, device='cuda')
        model.fit_predict(X)
        inertias.append(model.inertia_)

    # ひじ検出
    knee_locator = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    return optimal_k, inertias

def stratified_sample(X, labels, ratio=0.01, seed=None):
    """
    層化抽出によってXとlabelsからサブセットを取得する

    Args:
        X (Tensor): [N, D] の特徴ベクトル（CUDA対応）
        labels (Tensor): [N,] のラベル（int64, CUDA対応）
        ratio (float): 各クラスタから抽出する割合（例: 0.1）
        seed (int or None): ランダムシード

    Returns:
        X_sub (Tensor): 抽出された [M, D] データ（CUDA上）
        labels_sub (Tensor): 抽出された [M,] ラベル（CUDA上）
    """
    device = X.device
    if seed is not None:
        torch.manual_seed(seed)

    selected_indices = []

    for label in labels.unique():
        idx = (labels == label).nonzero(as_tuple=True)[0]
        n_select = max(1, int(len(idx) * ratio))  # 最低1件は残す
        perm = torch.randperm(len(idx), device=device)[:n_select]
        selected = idx[perm]
        selected_indices.append(selected)

    selected_indices = torch.cat(selected_indices)
    return X[selected_indices], labels[selected_indices]


def silhouette_per_cluster_euclidean(X, labels, chunk_size=5000):
    """
    クラスタごとの平均シルエットスコア（ユークリッド距離版）
    - X: (N, D) tensor (CUDA上)
    - labels: (N,) tensor (int64, CUDA上)
    - chunk_size: バッチサイズ（距離計算の分割単位）
    Returns:
        dict {label: mean silhouette score}
    """
    device = X.device
    N = X.size(0)
    labels = labels.to(device)

    clusters = {}
    for k in torch.unique(labels):
        mask = (labels == k)
        clusters[int(k.item())] = torch.where(mask)[0]

    a = torch.zeros(N, device=device)
    b = torch.full((N,), float('inf'), device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]  # (B, D)
        label_chunk = labels[start:end]
        B = end - start

        # ユークリッド距離計算
        dist_chunk = torch.cdist(X_chunk, X, p=2)  # (B, N)

        for i in range(B):
            idx = start + i
            label_i = int(label_chunk[i].item())
            idx_self = clusters[label_i]

            # a(i): 同クラスタ内平均距離（自分除く）
            if len(idx_self) > 1:
                d_same = dist_chunk[i, idx_self]
                a[idx] = (d_same.sum() - dist_chunk[i, idx]) / (len(idx_self) - 1)
            else:
                a[idx] = 0.0

            # b(i): 他クラスタの平均距離の最小値
            for other_k, idx_other in clusters.items():
                if other_k == label_i:
                    continue
                d_other = dist_chunk[i, idx_other]
                b[idx] = torch.minimum(b[idx], d_other.mean())

    max_ab = torch.maximum(a, b)
    s_all = (b - a) / (max_ab + 1e-8)
    silhouette_by_cluster = {}
    for k, idxs in clusters.items():
        silhouette_by_cluster[k] = s_all[idxs].mean().item()

    return silhouette_by_cluster


def silhouette_per_cluster_l2_normalized(X, labels, chunk_size=1000):
    """
    クラスタごとの平均シルエットスコア（L2正規化済みデータ前提）
    - X: (N, D) tensor (正規化済み, CUDA上)
    - labels: (N,) tensor (int64, CUDA上)
    - chunk_size: 分割単位
    Returns:
        dict {label: mean silhouette score for that cluster}
    """
    device = X.device
    N = X.size(0)
    labels = labels.to(device)

    # クラスタごとのインデックス
    clusters = {}
    for k in torch.unique(labels):
        mask = (labels == k)
        clusters[int(k.item())] = torch.where(mask)[0]

    a = torch.zeros(N, device=device)
    b = torch.full((N,), float('inf'), device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]  # shape: (B, D)
        label_chunk = labels[start:end]
        B = end - start

        # cosine距離 = 1 - 類似度（L2正規化済み前提）
        sim_chunk = X_chunk @ X.T
        dist_chunk = 1.0 - sim_chunk

        for i in range(B):
            idx = start + i
            label_i = int(label_chunk[i].item())
            idx_self = clusters[label_i]

            # a(i)
            if len(idx_self) > 1:
                d_same = dist_chunk[i, idx_self]
                a[idx] = (d_same.sum() - dist_chunk[i, idx]) / (len(idx_self) - 1)
            else:
                a[idx] = 0.0

            # b(i)
            for other_k, idx_other in clusters.items():
                if other_k == label_i:
                    continue
                d_other = dist_chunk[i, idx_other]
                b[idx] = torch.minimum(b[idx], d_other.mean())

    # 全体スコア
    max_ab = torch.maximum(a, b)
    s_all = (b - a) / (max_ab + 1e-8)

    # クラスタごとに平均スコア集計
    silhouette_by_cluster = {}
    for k, idxs in clusters.items():
        scores = s_all[idxs]
        silhouette_by_cluster[k] = scores.mean().item()

    return silhouette_by_cluster


