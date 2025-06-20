## ✅ データ前処理


1. **強度から反射率への変換**
   観測した強度データ $I(\lambda)$ を基準標準物質の強度 $I_{\text{ref}}(\lambda)$ で割ることで反射率 $R(\lambda)$ を算出しました：

$$
R = \frac{(I_{3.0} - D_{3.0}) / 3}{(W_{2.5} - D_{2.5}) / 2.5}
$$

2. **木材マスクの作成とスペクトル抽出**
   ハイパースペクトル画像から木材部分を抽出するためにマスク処理を行い、木材領域のスペクトルを取得しました。

3. **SNV処理（Standard Normal Variate）**
   各スペクトル $x$ に対して、平均値 $\bar{x}$ と不偏標準偏差 $s_x$ を用い、以下のように正規化しました：

$$
x_{\mathrm{SNV}} = \frac{x - \bar{x}}{s_x}
$$

---

## 🔍 主成分分析 （PCA）

累積寄与率80%以上となる最小の $k$ を選択

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
mm = MinMaxScaler()
pca = PCA()

pca.fit(hinoki_snv)
cvr = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cvr >= 0.80) + 1
```
<img src='Image\PCA_CVR.png'><br>

80%以上の累積寄与率を達成する最小次元数: $k=11$

```python
n_dim = 11
pca = PCA(n_components=n_dim)

hinoki_pca = pca.fit_transform(hinoki_snv)
hinoki_pca_mm = mm.fit_transform(hinoki_pca)
```

要約された11個の変数に対して0~1に正規化し、以下のように可視化した。

<table>
  <tr>
    <th>PC1</th>
    <th>PC2</th>
    <th>PC3</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA1.png" width="400"></td>
    <td><img src="Image/hinoki_PCA2.png" width="400"></td>
    <td><img src="Image/hinoki_PCA3.png" width="400"></td>
  </tr>
  <tr>
    <th>PC4</th>
    <th>PC5</th>
    <th>PC6</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA4.png" width="400"></td>
    <td><img src="Image/hinoki_PCA5.png" width="400"></td>
    <td><img src="Image/hinoki_PCA6.png" width="400"></td>
  </tr>
  <tr>
    <th>PC7</th>
    <th>PC8</th>
    <th>PC9</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA7.png" width="400"></td>
    <td><img src="Image/hinoki_PCA8.png" width="400"></td>
    <td><img src="Image/hinoki_PCA9.png" width="400"></td>
  </tr>
  <tr>
    <th>PC10</th>
    <th>PC11</th>
    <th></th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA10.png" width="400"></td>
    <td><img src="Image/hinoki_PCA11.png" width="400"></td>
    <td></td>
  </tr>
</table>

### PCA Loading Plot

<table>
  <tr>
    <th>PC1 Loading</th>
    <th>PC2 Loading</th>
    <th>PC3 Loading</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA1_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA2_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA3_loading.png" width="400"></td>
  </tr>
  <tr>
    <th>PC4 Loading</th>
    <th>PC5 Loading</th>
    <th>PC6 Loading</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA4_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA5_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA6_loading.png" width="400"></td>
  </tr>
  <tr>
    <th>PC7 Loading</th>
    <th>PC8 Loading</th>
    <th>PC9 Loading</th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA7_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA8_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA9_loading.png" width="400"></td>
  </tr>
  <tr>
    <th>PC10 Loading</th>
    <th>PC11 Loading</th>
    <th></th>
  </tr>
  <tr>
    <td><img src="Image/hinoki_PCA10_loading.png" width="400"></td>
    <td><img src="Image/hinoki_PCA11_loading.png" width="400"></td>
    <td></td>
  </tr>
</table>

### ❓ 試料のふち（PC1で顕著にみられる）で何が起こっている？ 背景の情報が木材試料のスペクトルに影響している気がする。
このノイズに対して効果的な対処法はありますか？
