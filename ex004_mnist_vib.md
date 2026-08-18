# 実験004: Deep Variational Information Bottleneck による MNIST 分類

## 1. 目的

本実験の目的は，実験001の分類器と実験003の確率的 Encoder を組み合わせ，Deep Variational Information Bottleneck（Deep VIB）により MNIST を分類することである．ラベル $Y$ に関する情報 $I(Z; Y)$ を保ちつつ入力 $X$ に関する情報 $I(Z; X)$ を圧縮する枠組みを，交差エントロピーと KL ダイバージェンスの加重和として実装する．

## 2. 手法

### 2.1 モデル

入力は $28 \times 28$ のグレースケール画像 $x$ である．Encoder は実験003と同様，$q(z|x)=\mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ のパラメータ $\mu$ と $\log\sigma^2$（各 32 次元）を出力する．

Encoder:

$$
h_1 = \mathrm{ReLU}(W_1^{(e)}\tilde{x} + b_1^{(e)}) \in \mathbb{R}^{256}
$$

$$
h_2 = \mathrm{ReLU}(W_2^{(e)} h_1 + b_2^{(e)}) \in \mathbb{R}^{128}
$$

$$
\mu = W_\mu h_2 + b_\mu \in \mathbb{R}^{32},\quad
\log\sigma^2 = W_{\log\sigma^2} h_2 + b_{\log\sigma^2} \in \mathbb{R}^{32}
$$

Reparameterization Trick により，

$$
z = \mu + \epsilon \odot \exp(0.5\log\sigma^2),\quad \epsilon \sim \mathcal{N}(0, I)
$$

Classifier は潜在ベクトル $z$ から 10 クラスのロジットを計算する．実験003の Decoder（画像再構成）を分類ヘッドへ置き換えた構造である．

$$
\ell = W_{\mathrm{out}} z + b_{\mathrm{out}} \in \mathbb{R}^{10}
$$

予測クラスは $\hat{y} = \arg\max_k \ell_k$ である．可視化用の `predict` はサンプリングを行わず，$\mu$ を分類器へ渡す．

### 2.2 誤差関数

Information Bottleneck の目的は次のトレードオフである．

$$
\min I(Z; X) - \beta^{-1} I(Z; Y)
$$

Deep VIB ではこれを次の損失で近似する．

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \beta \mathcal{L}_{\mathrm{KL}}
$$

交差エントロピー $\mathcal{L}_{\mathrm{CE}}$ は $I(Z; Y)$ の変分下界の負に対応する．ミニバッチサイズを $N$，サンプル $n$ の正解クラスを $y_n$，クラス $k$ のロジットを $\ell_{n,k}$ として，

$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N}\sum_{n=1}^{N}\left(\ell_{n,y_n} - \log\sum_{k=1}^{10}\exp(\ell_{n,k})\right)
$$

KL 項は $q(z|x)$ と事前分布 $p(z)=\mathcal{N}(0,I)$ の KL ダイバージェンスであり，$I(Z; X)$ の上界である．

$$
\mathcal{L}_{\mathrm{KL}} = -\frac{1}{2N}\sum_{n=1}^{N}\sum_{j=1}^{J}\left(1 + \log\sigma_{n,j}^2 - \mu_{n,j}^2 - \sigma_{n,j}^2\right)
$$

ここで $J=32$ は潜在次元，$\beta=0.001$ は圧縮の強さを決める係数である．

### 2.3 評価指標

評価指標は Accuracy と KL である．Loss は `iteration` 内で計算するため，`metrics_func` では Accuracy と KL のみを返す．最良モデルの選択基準は検証 Accuracy の最大値である．

$$
\mathrm{Accuracy} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}[\hat{y}_n = y_n]
$$

## 3. 実験条件

| 項目 | 設定 |
|:---|:---|
| データセット | MNIST（学習 60,000 枚，テスト 10,000 枚） |
| 前処理 | `ToTensor()`（画素値を $[0, 1]$ へ変換） |
| モデル | 全結合 Deep VIB（Encoder 784-256-128-($\mu$,$\log\sigma^2$)，Classifier 32-10） |
| 潜在次元 | $32$ |
| $\beta$ | $0.001$ |
| 最適化手法 | Adam |
| 学習率 | $0.001$ |
| バッチサイズ | $128$ |
| エポック数 | $10$（0 エポック目は初期値評価） |
| シード | $0$（サンプルのため 1 つ） |
| 最良モデルの基準 | 検証 Accuracy の最大値 |

結果の保存先は `outputs/ex004_mnist_vib/Adam/0.001_128/0/` である．


## 4. 実験結果

学習は完了した．0 エポック目の検証 Accuracy は $0.1023$ であり，10 クラスのチャンスレート近傍である．1 エポック目で検証 Accuracy は $0.9509$ まで上昇し，6 エポック目に最高値 $0.9785$ を記録した．

| epoch | train Loss | train acc | train KL | val Loss | val acc | val KL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2.4502 | 0.1014 | 0.0649 | 2.4494 | 0.1023 | 0.0650 |
| 1 | 0.4634 | 0.8799 | 66.30 | 0.2155 | 0.9509 | 54.23 |
| 6 | 0.0674 | 0.9891 | 29.51 | 0.1051 | **0.9785** | 28.80 |
| 10 | 0.0421 | 0.9948 | 23.33 | 0.1075 | 0.9780 | 22.18 |

最良検証 Accuracy は $0.9785$（epoch 6）である．実験001の全結合 NN（最良検証 Accuracy $0.9808$）とほぼ同水準である．KL は学習開始直後に上昇したのち緩やかに減少し，最終エポックでも $22.18$ を保った．実験003の $\beta$-VAE と同じ $\beta=0.001$ を用いる．

### 4.1 学習曲線

![学習 Loss](figures/ex004_mnist_vib/train_loss.png)

![検証 Loss](figures/ex004_mnist_vib/val_loss.png)

![学習 Accuracy](figures/ex004_mnist_vib/train_acc.png)

![検証 Accuracy](figures/ex004_mnist_vib/val_acc.png)

![学習 KL](figures/ex004_mnist_vib/train_kl.png)

![検証 KL](figures/ex004_mnist_vib/val_kl.png)

塗りつぶしは seed 間の標準偏差である．本実験は seed が 1 つであるため，区間幅は 0 である．

### 4.2 予測例

![予測の例](figures/ex004_mnist_vib/predictions.png)

検証バッチの数字 7, 2, 1, 0, 4, 1, 4, 9 はいずれも正解ラベルと一致した．

## 5. 考察

Deep VIB は VAE と同じ確率的 Encoder を用いるが，目的が再構成ではなく分類である．交差エントロピーが $I(Z; Y)$ を押し上げるため，$q(z|x)$ はラベル識別に必要な情報を潜在空間へ残す．その結果，KL は 0 へ潰れず，VAE で観測した Posterior Collapse を回避できた．

検証 Accuracy は実験001の決定的分類器とほぼ同等であり，32 次元の確率的ボトルネックを挟んでも識別性能が大きく落ちないことを示す．一方，最終エポックで学習 Accuracy は $0.9948$ まで上昇し，検証 Accuracy は $0.9780$ で頭打ちである．$\beta$ を大きくすると圧縮が強まり，過学習の抑制が期待できる．次段の 2 次元特徴マップ可視化では，この潜在空間がクラスごとに分離しているかを直接確認する．

## 6. まとめ

- 全結合 Deep VIB（潜在次元 32，$\beta=0.001$）により MNIST を分類し，最良検証 Accuracy $0.9785$ を得た．
- KL は最終エポックでも約 $22$ を保ち，VAE で生じた Posterior Collapse は観測されなかった．
- 分類性能は実験001の全結合 NN（$0.9808$）とほぼ同水準である．
- 結果は `outputs/ex004_mnist_vib/Adam/0.001_128/0/` に保存した．
