# Spatial Embedding Continuity Evaluation (DISSE vs. ELSA)

## 1. 目的

音源を同一距離に保ったまま方位角 (azimuth) または仰角 (elevation) を連続的に変化させ、物理的な方向の局所構造が音声埋め込み空間でどの程度保存されているかを評価する。DISSE の空間音声埋め込み `audio_space` と、ベースライン ELSA の `audio_elsa` を比較する。

評価観点は (1) 物理角度に基づく最近傍関係の保存（QNX(2)）、(2) 隣接する物理角度に対応する埋め込み距離の均一性（Sadj）である。円環図は定性的な補助評価として使用する。

## 2. 実験条件

設定の入口は [`configs/continuity_additional.toml`](configs/continuity_additional.toml)。条件の読み込み・検証は `continuity_experiment/config.py`、室内音響の校正は `continuity_experiment/acoustics.py` が担当する。

| 項目 | 設定 |
| --- | --- |
| モデル | DISSE / ELSA |
| 音源距離 | 1.5 m |
| Azimuth sweep | 0°, 5°, ..., 355°（72点） |
| Azimuth sweep 中の仰角 | 0°固定 |
| Elevation sweep | -45°, -40°, ..., 45°（19点） |
| Elevation sweep 中の方位角 | 0°固定 |
| small room | 6 × 5 × 3 m |
| medium room | 10 × 7.5 × 3 m |
| large room | 15 × 10 × 3 m |
| 目標 T30 | 0.0 / 0.3 / 0.6 s |
| T30 校正許容差 | ±0.02 s |
| 校正最大反復数 | 16 |

受音点は直方体の中央。音源座標は受音点 `c` を中心として

```text
x = c_x + r cos(elevation) cos(azimuth)
y = c_y + r cos(elevation) sin(azimuth)
z = c_z + r sin(elevation)
```

で求める（`r=1.5 m`）。よって方位角・仰角とも0°の音源は、受音点から水平の +x 方向に1.5 mの位置にある。

1音源あたり **2モデル × 2軸 × 3部屋 × 3残響条件 = 36ジョブ**。10音源なら360ジョブとなる。

### RIR と前処理

室内インパルス応答は `pyroomacoustics` でシミュレーションする。T30>0では方位角0°・仰角0°のRIRを基準としてT30を推定し、吸音率を校正する。T30はFOAのW成分に相当するRIRのSchroeder減衰曲線の -5～-35 dB 区間から推定する。**全方向のT30が厳密に一定という保証ではない。**

T30=0は `alpha=1.0, max_order=0` の直接音条件。dry音声とA-format RIRを畳み込み、FOA（W,Y,Z,X）へ変換する。FOAの生成周波数は48 kHz、Intensity Vectorの算出用信号は16 kHzで、STFTは `n_fft=400, hop_length=100`。

直方体の壁までの距離は方向によって変わるため、残響条件での方向スイープには反射音の方向依存性も含まれる。直接音のみの条件と分けて解釈する。

## 3. 評価指標

埋め込み間の距離はコサイン距離

```text
d(i,j) = 1 - cosine_similarity(z_i, z_j)
```

を使う。

### QNX(2): 物理近傍の保存

各点について、物理角度が最も近い2点の集合 `N_phys(i)` と、埋め込み距離が最も近い2点の集合 `N_emb(i)` の重複率を計算し、全点で平均する。

```text
QNX(2) = mean_i( |N_phys(i) ∩ N_emb(i)| / 2 )
```

0～1の範囲で **高いほど良い**。Azimuthは0°と355°を隣接する周期空間として扱う。Elevationは端点を接続しない。

### Sadj: 隣接距離の均一性

物理角度順の隣接埋め込み距離を `d_k`、その平均を `mean(d)` として

```text
Sadj = sum_k((d_k - mean(d))^2) / sum_k(d_k^2)
```

を計算する。**低いほど隣接距離のばらつきが小さい**。Azimuthでは最後の点から最初の点への辺を含むが、Elevationでは含まない。分母が数値的に0の場合は未定義（JSONの `null`）となる。

QNX(2)は近傍集合の一致、Sadjは隣接距離の均一性を測るため、どちらか一方のみで「空間構造が適切」と判断しない。埋め込みが潰れている場合にも注意する。

実装は `evaluate_disse_continuity.py` と `evaluate_elsa_continuity.py`。

## 4. 実行手順

Python環境と、必要な依存パッケージ（PyTorch、torchaudio、NumPy、pyroomacoustics、matplotlib等）を用意する。設定ファイルに記載した音源・DISSEチェックポイント・ELSAチェックポイント・ELSAのコードツリーがローカルに存在する必要がある。

まず `configs/continuity_additional.toml` 内のパスを確認する。

```bash
# 実行予定のジョブのみ確認（RIR生成を行わない）
python run_continuity_experiment.py --dry-run

# 1ジョブだけ試す
python run_continuity_experiment.py --limit-jobs 1

# 全条件
python run_continuity_experiment.py

# DISSEのみ
python run_continuity_experiment.py --models disse
```

別の設定ファイルを使用する場合は `--config path/to/your.toml` を指定する。

複数音源を評価する場合は `audio_sources` を増やし、`audio_source_mode = "mean"` に設定する。デフォルトのTOMLは **107280.mp3 の1音源**、`audio_source_mode="single"`、出力先 `results/continuity_additional_3x3` である。

## 5. 出力

```text
<output_root>/
├── calibration.json
├── summary.csv
├── summary_mean.csv                      # meanモードで対象音源が揃った場合
└── <source_id>/
    ├── disse/
    │   ├── azimuth/<condition>/
    │   │   ├── run_spec.json
    │   │   ├── continuous_azimuth_embeddings.pt
    │   │   └── continuity_metrics_audio_space.json
    │   └── elevation/<condition>/...
    └── elsa/
        ├── azimuth/<condition>/
        │   ├── run_spec.json
        │   ├── elsa_embeddings.pt
        │   └── metrics.json
        └── elevation/<condition>/...
```

`summary.csv` は音源・モデル・軸・部屋・目標T30・基準位置での実測T30・QNX(2)・Sadj・埋め込みおよび指標ファイルのパスを格納する。書き込みは `continuity_experiment/jobs.py:update_summary()`。

`summary_mean.csv` は `update_mean_summary()` で計算し、同じモデル・軸・部屋・T30のグループについて **指定音源がすべて揃った場合にのみ** QNX(2) と Sadj の音源間平均を保存する。

`calibration.json` は室内ごとの校正結果、`run_spec.json` は各ジョブの実際の条件を保持する。条件が既存の `run_spec.json` と異なる場合は安全のため実行が停止される。

## 6. コサイン距離による円環可視化

[`plot_embedding_cosine_polar.py`](plot_embedding_cosine_polar.py) に次の3種類が実装されている。

| モード | 表示 |
| --- | --- |
| `radial` | 0°基準埋め込みからの距離を半径に反映 |
| `unit` | 物理角度で単位円上に配置し、距離を色で表示 |
| `distance-circle` | 隣接埋め込み間の距離に比例した間隔で単位円上に配置 |

`distance-circle` は、隣接コサイン距離 `d_k` を、Azimuthでは円周全体 `2π`、Elevationでは物理的な仰角の範囲に正規化して累積配置する。点の色は元の物理角度、辺の色は正規化前の隣接コサイン距離を示す。

**注意：PCA/t-SNEのような埋め込みの2次元射影ではない。** 隣接関係だけをもとに配置し、距離全体を一定の弧長に正規化するため、絶対的な距離のスケールや大域的な距離構造は表せない。

```bash
# 個別例
python plot_embedding_cosine_polar.py \
  --emb-pt results/continuity_additional_3x3_10sources/107280/disse/azimuth/small_t30_0p3s/continuous_azimuth_embeddings.pt \
  --axis azimuth --key audio_space --plot-mode distance-circle

# 保存済みの全条件から図を生成
python generate_distance_circle_plots.py \
  --input-root results/continuity_additional_3x3_10sources \
  --output-root results/visualizations/cosine_distance_circle_10sources

# 特定の条件群のみ
python generate_distance_circle_plots.py --model disse --axis azimuth --source 107280
```

一括可視化は各条件の `distance_circle.png` と一覧 `manifest.csv` を生成する。

## 7. 結果の解釈と再現性

比較では、同じ音源・部屋・T30・方向軸のDISSEとELSAを並べる。定量的な比較は `summary.csv` または `summary_mean.csv`、可視化は補助資料として扱う。音源差・部屋差・残響差・azimuth/elevation差を混同しない。

既存の `results/continuity_additional_3x3_10sources/` は10音源用の実験結果だが、**現在リポジトリ上のデフォルトTOMLは1音源用**である。10音源実験の正確な再現には、実際に使用した音源リスト、各 `run_spec.json` と `calibration.json`、チェックポイントを照合すること。現在のTOMLをそのまま「10音源実験の完全な設定」として扱わない。

## 8. 関連コード

- `configs/continuity_additional.toml` — 条件
- `run_continuity_experiment.py` — 実験の入口
- `continuity_experiment/config.py` — 設定検証
- `continuity_experiment/acoustics.py` — T30校正
- `continuity_experiment/commands.py` — モデルごとの実行コマンド
- `continuity_experiment/jobs.py` — ジョブ・summary集計
- `continuous_azimuth_heatmap.py`, `continuous_elevation_heatmap.py` — DISSE埋め込み
- `evaluate_disse_continuity.py`, `evaluate_elsa_continuity.py` — 評価指標
- `plot_embedding_cosine_polar.py`, `generate_distance_circle_plots.py` — 可視化
