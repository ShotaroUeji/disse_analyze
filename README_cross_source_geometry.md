# DISSE / ELSA cross-source geometry

このディレクトリのコードは、保存済みの連続角度埋め込みだけを使って、音源をまたいだ空間表現の整合性を比較する。音声生成、RIR生成、チェックポイント推論、再学習は行わない。

## 入力

既定の入力レイアウトは次のとおり。

```text
results/continuity_additional_3x3_10sources/<source>/<model>/<axis>/<condition>/*embeddings.pt
```

DISSEは`embeddings["audio_space"]`、ELSAは`embeddings["audio_elsa"]`を使う。読み込み時に角度グリッド、埋め込み次元、有限値、ゼロベクトル、音源間の対応順序を検証する。`.pt`は`torch.load(..., weights_only=True)`で読み込む。

## 最小実験

medium・T30=0.3秒・方位角を実行する。

```bash
export MPLCONFIGDIR=/tmp/delsa-matplotlib
PYTHONPATH=. ./.venv/bin/python -m cross_source_geometry.run_analysis \
  --input-root results/continuity_additional_3x3_10sources \
  --axis azimuth --room medium --t30 0.3 \
  --models elsa disse \
  --experiments distance retrieval rsa \
  --output-root results/cross_source_geometry
```

全条件や仰角は、例えば次のように実行できる。

```bash
PYTHONPATH=. ./.venv/bin/python -m cross_source_geometry.run_analysis \
  --input-root results/continuity_additional_3x3_10sources \
  --axis both --room all --t30 all \
  --models elsa disse \
  --experiments distance retrieval rsa \
  --output-root results/cross_source_geometry_all
```

`azimuth`は0〜355°の周期として扱い、`elevation`は開いた経路として扱う。

## 指標

### Cross-source distance

音源A、Bの同じ条件の埋め込みに対して

\[
D_{A,B}(i,j)=1-\cos(z_{A,i},z_{B,j})
\]

を計算する。

- `same_angle_mean`: 対角成分の平均
- `different_angle_mean`: 非対角成分の平均
- `separation`: `different_angle_mean - same_angle_mean`
- `cross_source_distance_by_delta.csv`: 角度差δごとの平均距離

45ペアの距離行列を保存し、平均ヒートマップは両モデルともカラースケール0〜2で描画する。

### Retrieval

- `pairwise_cross_source`: A→B、B→Aの72方向検索
- `leave_one_source_out`: 1音源をクエリ、残り9音源×72方向をギャラリー
- `within_source_physical_neighbor`: 同一音源で自己項目を除き、物理的に最も近い角度を正例とする比較タスク

各CSVにR@1、R@5、MedR、Circular MAE、複数正例を考慮したchance levelを保存する。

### RSA

各音源内の72×72距離行列の上三角をベクトル化し、音源ペア間のSpearman相関を計算する。RSAは距離構造の類似性を示すが、直交回転そのものの証明ではない。

### Procrustes

必要な場合だけ`--experiments procrustes`を指定する。同じ角度の対応点を偶数・奇数インデックスに分け、訓練角度で直交変換を推定し、評価角度で変換前後のコサイン距離を比較する。高次元512次元に対して72点なので、結果は過適合に注意して扱う。

## テスト

合成埋め込みで、不変埋め込み、直交回転、非線形変形、方位角の周期性、ファイル異常を検証する。

```bash
PYTHONPATH=. ./.venv/bin/python tests/test_cross_source_geometry.py
```

## 出力

`results/cross_source_geometry/`に、条件別の距離行列、検索結果、RSA行列、PNG/PDF、CSV、`report.md`を保存する。`input_manifest.csv`には実際に読み込んだファイル、キー、点数、埋め込み次元、ノルム範囲を記録する。45ペアは独立標本とみなさず、距離指標の要約には音源ID単位のbootstrapを併記する。
