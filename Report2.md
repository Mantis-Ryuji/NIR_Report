## 🥅 目的

本研究では、**近赤外ハイパースペクトル画像（NIR-HSI）**を用いて、木材の**非破壊による劣化診断手法**の確立を目指します。近赤外スペクトルは微細な化学変化を高次元で捉えられる一方、測定ノイズや劣化に関係のないノイズに対しても敏感であるため、**劣化に敏感かつノイズに頑健な特徴抽出**が求められます。<br>

本手法では、従来の線形次元削減に加えて、**畳み込みオートエンコーダ（CAE）**および**コントラスト学習**を組み合わせた特徴抽出を行い、**UMAPによる非線形圧縮**と**HDBSCANによるクラスタリング**を通じて、潜在空間上での劣化状態の分離性と構造性を評価します。

## 📖 分析のフロー

学習データとして `hinoki_snv`（反射率をSNV処理したもの）を用いた。<br>
また、`UMAP` 投影前には各変数において標準化した（`StandardScaler`）

| 構成                      | 分析フロー                                                                                                    | 目的                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **PCA → UMAP**          | `hinoki_snv (256D) → hinoki_pca (11D) → hinoki_pca_sc (11D) → UMAP (2D) → HDBSCAN`                       | 線形圧縮によるノイズ低減と非線形構造抽出の組み合わせ    |
| **CAE → UMAP**          | `hinoki_snv (256D) → hinoki_cae (16D) → hinoki_cae_sc (16D) → UMAP (2D) → HDBSCAN`                       | 再構成誤差を指標としたデータ主導型の特徴抽出        |
| **CAE+Contrast → UMAP** | `hinoki_snv (256D) → hinoki_contrastivecae (16D) → hinoki_contrastivecae_sc (16D) → UMAP (2D) → HDBSCAN` | 再構成性と識別性を両立した特徴抽出による劣化クラスタの強調 |


## 🎯 各構成の意味と役割

| 構成                      | 解釈・ねらい                             |
| ----------------------- | ---------------------------------- |
| **PCA → UMAP**          | 線形圧縮で高次元ノイズを除去した上で、局所構造をUMAPで可視化   |
| **CAE → UMAP**          | オートエンコーダによる圧縮が、構造保持・ノイズ除去にどれほど有効か    |
| **CAE+Contrast → UMAP** | ノイズを識別するように学習された潜在空間が、クラスタ構造を強化するか |

---

## 📚 潜在空間の可視化（UMAP）& HDBSCAN によるクラスタリング
```python
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
hdbscan = HDBSCAN(min_cluster_size=1000, min_samples=100)
```
<table>
  <tr>
    <th>PCA → UMAP</th>
    <th>CAE → UMAP</th>
    <th>CAE+Contrast → UMAP</th>
  </tr>
  <tr>
    <td><img src="Image/PCA_UMAP.png" width="400"></td>
    <td><img src="Image/CAE_latent.png" width="400"></td>
    <td><img src="Image/ContrastiveCAE_latent.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="Image/PCA_label.png" width="400"></td>
    <td><img src="Image/CAE_latent_labels.png" width="400"></td>
    <td><img src="Image/ContrastiveCAE_latent_labels.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="Image/PCA_persistence.png" width="400"></td>
    <td><img src="Image/CAE_persistence.png" width="400"></td>
    <td><img src="Image/ContrastiveCAE_persistence.png" width="400"></td>
  </tr>
</table>

`cluster_persistence` は、HDBSCAN（Hierarchical Density-Based Spatial Clustering of Applications with Noise）に特有の**クラスタの「安定性」指標**です。

---

## 📖 解釈

### クラスタリング結果 に対して：
  1. **クラスタ分布の可視化**
  2. **Reflectance**
  3. **二次微分（SG法）**

以上を **構成間で比較し、解釈** する

### 🔍 1. 各クラスタの分布

<table>
  <tr>
    <th>PCA → UMAP</th>
    <th>CAE → UMAP</th>
    <th>CAE+Contrast → UMAP</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA_label.png" width="400"></td>
    <td><img src="Image/hinoki_CAE_label.png" width="400"></td>
    <td><img src="Image/hinoki_ContrastiveCAE_label.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="Image/PCA_reflectance.png" width="400"></td>
    <td><img src="Image/CAE_reflectance.png" width="400"></td>
    <td><img src="Image/ContrastiveCAE_reflectance.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="Image/PCA_reflectance_dd.png" width="400"></td>
    <td><img src="Image/CAE_reflectance_dd.png" width="400"></td>
    <td><img src="Image/ContrastiveCAE_reflectance_dd.png" width="400"></td>
  </tr>
</table>

**SG法の設定**
```python
from scipy.signal import savgol_filter

window_length = 7 
polyorder = 2
deriv_order = 2
```
