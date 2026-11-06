# Cross-source geometry report

## 実行概要

実行コマンド: `/home/takamichi-lab-pc09/DELSA/cross_source_geometry/run_analysis.py --input-root results/continuity_additional_3x3_10sources --axis azimuth --room medium --t30 0.3 --models elsa disse --experiments distance retrieval rsa --output-root results/cross_source_geometry`

保存済みのDISSE/ELSA埋め込みだけを読み込み、音声生成・RIR生成・モデル推論は行っていない。
方位角は周期的、仰角は非周期的な変数として扱った。45音源ペアは独立標本とみなさず、必要な要約では音源ID単位のbootstrapを使用する。

## 条件別の主要結果

| axis | room | T30 [s] | model | sources | dim | D_same | D_diff | Δ | Retrieval R@1 (LOSO) | RSA mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| azimuth | medium | 0.3 | disse | 10 | 512 | 0.2204 | 0.6082 | 0.3878 | 0.3444 | 0.9150 |
| azimuth | medium | 0.3 | elsa | 10 | 512 | 0.6268 | 0.7988 | 0.1719 | 0.1500 | 0.9107 |

## 解釈上の注意

- D_same が小さいことは、同じ角度の音源間埋め込みが近いことを示す。
- RSA は音源内の距離構造の一致度であり、埋め込み座標の位置合わせそのものを証明しない。
- 高いRSAと大きいCross-source距離が同時に得られた場合、位置・向きのずれの可能性は示唆されるが、回転を断定しない。Procrustesは追加解析として扱う。
- Retrievalの正例は同一物理角度の別音源候補であり、chance levelを各CSVに併記した。

## 出力

条件別ディレクトリのCSVとPNG/PDF、`input_manifest.csv`、`aggregate_results.csv`を参照する。
